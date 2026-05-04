"""Training loop for the binary CodeBERT classifier.

Design choices
--------------

* **AdamW** with the standard "decay weights but not biases or LayerNorm"
  parameter grouping. This is the recipe used in the original BERT/RoBERTa
  fine-tuning code and is what every transformers-based classifier uses
  unless they have a specific reason to deviate.
* **Linear warmup followed by linear decay** of the learning rate -- the
  HuggingFace default schedule, again the standard for transformer
  fine-tuning.
* **Per-epoch validation** with optional early stopping on validation loss.
  The best-by-val-loss checkpoint is what we report on -- this prevents
  reporting test-set numbers from a model that overfit on the last epoch.
* **Gradient clipping** at norm 1.0 (CodeBERT paper recipe) for stability.

The loop returns a :class:`TrainingResult` that captures the best checkpoint
path, the per-epoch metric history, and the training config used. Scripts
serialize that into ``training_history.json`` next to the checkpoint.
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup

from ai_code_detector import config
from ai_code_detector.data.torch_dataset import batch_to_device
from ai_code_detector.models.classifier import CodeBertBinaryClassifier
from ai_code_detector.training.checkpoint import TrainingConfig, save_checkpoint
from ai_code_detector.training.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(prefer: str | None = None) -> torch.device:
    """Pick the best available compute device.

    Order of preference: explicit ``prefer`` argument, then CUDA, then MPS,
    then CPU. We log the selection so logs make the run reproducible.
    """
    if prefer is not None:
        device = torch.device(prefer)
        logger.info("Using user-requested device: %s", device)
        return device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Auto-selected device: %s", device)
    return device


# ---------------------------------------------------------------------------
# Optimizer / scheduler builders
# ---------------------------------------------------------------------------
_NO_DECAY_KEYWORDS: tuple[str, ...] = ("bias", "LayerNorm.weight", "layer_norm.weight")


def build_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """Build AdamW with the standard transformer fine-tuning parameter grouping.

    Weights of biases and LayerNorm parameters are excluded from weight decay,
    which slightly improves generalization in the BERT/RoBERTa family.
    """
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(keyword in name for keyword in _NO_DECAY_KEYWORDS):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def build_scheduler(
    optimizer: AdamW,
    *,
    num_training_steps: int,
    warmup_ratio: float,
):
    """Build the linear-warmup-then-linear-decay LR schedule."""
    warmup_steps = max(1, math.floor(num_training_steps * warmup_ratio))
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class EpochRecord:
    """Per-epoch summary written to training_history.json."""

    epoch: int
    train_loss: float
    val_loss: float
    val_metrics: dict
    learning_rate: float
    seconds: float


@dataclass(slots=True)
class TrainingResult:
    """Aggregated outputs of one :func:`train` call."""

    history: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = math.inf
    best_val_metrics: ClassificationMetrics | None = None
    checkpoint_dir: Path | None = None

    def history_as_dicts(self) -> list[dict]:
        """Return the history as a JSON-friendly list of dicts."""
        return [
            {
                "epoch": r.epoch,
                "train_loss": r.train_loss,
                "val_loss": r.val_loss,
                "val_metrics": r.val_metrics,
                "learning_rate": r.learning_rate,
                "seconds": r.seconds,
            }
            for r in self.history
        ]


@torch.no_grad()
def evaluate_loss_and_metrics(
    model: CodeBertBinaryClassifier,
    dataloader: Iterable[dict],
    device: torch.device,
    *,
    threshold: float = config.DECISION_THRESHOLD,
) -> tuple[float, ClassificationMetrics]:
    """Run one full pass over ``dataloader`` and return (avg_loss, metrics)."""
    model.eval()

    total_loss = 0.0
    total_examples = 0
    all_probs: list[float] = []
    all_labels: list[int] = []

    for batch in dataloader:
        batch = batch_to_device(batch, device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        batch_size = batch["input_ids"].size(0)
        # out.loss is mean over the batch; weight by batch size to get a true
        # dataset-level mean even when the last batch is smaller.
        total_loss += float(out.loss.item()) * batch_size
        total_examples += batch_size
        all_probs.extend(torch.sigmoid(out.logits).detach().cpu().tolist())
        all_labels.extend(batch["labels"].detach().cpu().tolist())

    avg_loss = total_loss / max(total_examples, 1)
    metrics = compute_classification_metrics(
        y_true=all_labels,
        y_proba=all_probs,
        threshold=threshold,
    )
    return avg_loss, metrics


def train(
    *,
    model: CodeBertBinaryClassifier,
    tokenizer: PreTrainedTokenizerBase,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_config: TrainingConfig,
    checkpoint_dir: Path,
    device: torch.device,
) -> TrainingResult:
    """Fine-tune the classifier and persist the best-by-val-loss checkpoint.

    Args:
        model: An (untrained) :class:`CodeBertBinaryClassifier`.
        tokenizer: The matching tokenizer (saved alongside the checkpoint).
        train_loader: Training DataLoader (shuffled).
        val_loader: Validation DataLoader (not shuffled).
        training_config: Hyperparameters (also persisted with the checkpoint).
        checkpoint_dir: Where to write the best-by-val-loss checkpoint.
        device: Compute device.

    Returns:
        A :class:`TrainingResult` summarizing the run.
    """
    set_seed(training_config.random_seed)
    model.to(device)

    num_training_steps = len(train_loader) * training_config.num_epochs
    optimizer = build_optimizer(
        model,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        num_training_steps=num_training_steps,
        warmup_ratio=training_config.warmup_ratio,
    )

    logger.info(
        "Trainable params: %s | total steps: %d | warmup_ratio=%.2f",
        f"{model.num_parameters(trainable_only=True):,}",
        num_training_steps,
        training_config.warmup_ratio,
    )

    result = TrainingResult()
    epochs_without_improvement = 0

    for epoch in range(1, training_config.num_epochs + 1):
        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        running_examples = 0

        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{training_config.num_epochs} train",
            leave=False,
        )
        for batch in progress:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out.loss
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=training_config.gradient_clip_norm
            )
            optimizer.step()
            scheduler.step()

            batch_size = batch["input_ids"].size(0)
            running_loss += float(loss.item()) * batch_size
            running_examples += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_loss = running_loss / max(running_examples, 1)

        val_loss, val_metrics = evaluate_loss_and_metrics(
            model,
            val_loader,
            device,
            threshold=training_config.decision_threshold,
        )

        epoch_seconds = time.time() - epoch_start
        record = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_metrics=val_metrics.to_dict(),
            learning_rate=scheduler.get_last_lr()[0],
            seconds=epoch_seconds,
        )
        result.history.append(record)
        logger.info(
            "epoch %d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_f1=%.4f | val_auc=%.4f | %.1fs",
            epoch,
            train_loss,
            val_loss,
            val_metrics.accuracy,
            val_metrics.macro_f1,
            val_metrics.auc_roc,
            epoch_seconds,
        )

        # Track / save best-by-val-loss.
        if val_loss < result.best_val_loss - 1e-6:
            result.best_val_loss = val_loss
            result.best_epoch = epoch
            result.best_val_metrics = val_metrics
            result.checkpoint_dir = checkpoint_dir
            save_checkpoint(
                checkpoint_dir,
                model=model,
                tokenizer=tokenizer,
                training_config=training_config,
                training_history=result.history_as_dicts(),
            )
            epochs_without_improvement = 0
            logger.info("New best val_loss=%.4f -> checkpoint updated.", val_loss)
        else:
            epochs_without_improvement += 1
            logger.info(
                "No improvement for %d epoch(s) (best val_loss=%.4f at epoch %d).",
                epochs_without_improvement,
                result.best_val_loss,
                result.best_epoch,
            )
            if epochs_without_improvement >= training_config.early_stop_patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d exhausted).",
                    epoch,
                    training_config.early_stop_patience,
                )
                break

    # Final write of training_history alongside the (already-saved) best
    # checkpoint so the JSON includes every epoch we actually ran.
    if result.checkpoint_dir is not None:
        (result.checkpoint_dir / "training_history.json").write_text(
            __import__("json").dumps(result.history_as_dicts(), indent=2),
            encoding="utf-8",
        )

    return result
