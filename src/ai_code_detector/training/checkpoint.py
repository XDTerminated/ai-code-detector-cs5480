"""Checkpoint serialization for trained classifiers.

Each checkpoint directory contains:

* ``model.safetensors``  -- model weights (state_dict).
* ``training_config.json`` -- hyperparameters used for the run.
* ``training_history.json`` -- per-epoch train/val loss + metrics, written
  by the training loop. Optional at load time.
* ``tokenizer/``         -- the HuggingFace tokenizer (so inference does not
  require an internet connection or knowledge of the original model name).

We intentionally avoid pickling the model object: ``safetensors`` is the
industry-standard, portable, secure format and round-trips cleanly across
torch versions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ai_code_detector import config
from ai_code_detector.models.classifier import CodeBertBinaryClassifier

logger = logging.getLogger(__name__)

_MODEL_FILENAME = "model.safetensors"
_TRAIN_CONFIG_FILENAME = "training_config.json"
_TRAIN_HISTORY_FILENAME = "training_history.json"
_TOKENIZER_DIRNAME = "tokenizer"


@dataclass(slots=True)
class TrainingConfig:
    """Hyperparameters captured for one training run.

    Persisted as JSON so any later evaluation, ablation, or report-writing
    step can reconstruct exactly what produced the checkpoint.
    """

    model_name: str = config.MODEL_NAME
    num_epochs: int = config.DEFAULT_NUM_EPOCHS
    batch_size: int = config.DEFAULT_BATCH_SIZE
    eval_batch_size: int = config.DEFAULT_EVAL_BATCH_SIZE
    learning_rate: float = config.DEFAULT_LEARNING_RATE
    weight_decay: float = config.DEFAULT_WEIGHT_DECAY
    warmup_ratio: float = config.DEFAULT_WARMUP_RATIO
    gradient_clip_norm: float = config.DEFAULT_GRADIENT_CLIP_NORM
    max_sequence_length: int = config.MAX_SEQUENCE_LENGTH
    classifier_dropout: float = config.CLASSIFIER_DROPOUT
    decision_threshold: float = config.DECISION_THRESHOLD
    early_stop_patience: int = config.DEFAULT_EARLY_STOP_PATIENCE
    random_seed: int = config.RANDOM_SEED
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def save_checkpoint(
    checkpoint_dir: Path,
    *,
    model: CodeBertBinaryClassifier,
    tokenizer: PreTrainedTokenizerBase,
    training_config: TrainingConfig,
    training_history: list[dict] | None = None,
) -> None:
    """Persist a trained model + tokenizer + config to ``checkpoint_dir``."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # State dict needs to be a flat {str: Tensor} mapping. We pull onto CPU
    # so checkpoints written from a GPU box load anywhere.
    state_dict = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    safe_save_file(state_dict, str(checkpoint_dir / _MODEL_FILENAME))

    (checkpoint_dir / _TRAIN_CONFIG_FILENAME).write_text(
        json.dumps(training_config.to_dict(), indent=2),
        encoding="utf-8",
    )
    if training_history is not None:
        (checkpoint_dir / _TRAIN_HISTORY_FILENAME).write_text(
            json.dumps(training_history, indent=2),
            encoding="utf-8",
        )

    tokenizer.save_pretrained(str(checkpoint_dir / _TOKENIZER_DIRNAME))
    logger.info("Saved checkpoint to %s", checkpoint_dir)


def load_training_config(checkpoint_dir: Path) -> TrainingConfig:
    """Load and return the :class:`TrainingConfig` for a checkpoint."""
    payload = json.loads((checkpoint_dir / _TRAIN_CONFIG_FILENAME).read_text(encoding="utf-8"))
    extra = payload.pop("extra", {})
    return TrainingConfig(**payload, extra=extra)


def load_training_history(checkpoint_dir: Path) -> list[dict]:
    """Return the per-epoch history list, or an empty list if not present."""
    path = checkpoint_dir / _TRAIN_HISTORY_FILENAME
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint(
    checkpoint_dir: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[CodeBertBinaryClassifier, PreTrainedTokenizerBase, TrainingConfig]:
    """Reconstruct the model, tokenizer, and training config from disk."""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    training_config = load_training_config(checkpoint_dir)

    model = CodeBertBinaryClassifier(
        model_name=training_config.model_name,
        dropout_prob=training_config.classifier_dropout,
    )
    state_dict = safe_load_file(str(checkpoint_dir / _MODEL_FILENAME), device=str(map_location))
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state_dict mismatch (missing={missing}, unexpected={unexpected})."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_dir / _TOKENIZER_DIRNAME), use_fast=True
    )
    logger.info("Loaded checkpoint from %s", checkpoint_dir)
    return model, tokenizer, training_config
