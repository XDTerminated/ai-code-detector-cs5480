"""Phase 3: fine-tune CodeBERT on the tokenized AI-vs-human Python dataset.

Run with::

    uv run python scripts/train.py

Reads the tokenized HuggingFace ``DatasetDict`` produced by
``scripts/tokenize_dataset.py`` from ``data/processed/tokenized/`` and
writes the best-by-val-loss checkpoint to ``models/baseline/``.

Hyperparameters default to the values in :mod:`ai_code_detector.config`
but every one is overridable on the command line, which makes this script
the workhorse for both the baseline run and any later ablation/sweep.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from ai_code_detector import config  # noqa: E402
from ai_code_detector.data.torch_dataset import (  # noqa: E402
    CodeClassificationDataset,
    build_dataloader,
)
from ai_code_detector.logging_utils import configure_logging  # noqa: E402
from ai_code_detector.models.classifier import CodeBertBinaryClassifier  # noqa: E402
from ai_code_detector.training.checkpoint import TrainingConfig  # noqa: E402
from ai_code_detector.training.loop import (  # noqa: E402
    select_device,
    set_seed,
    train,
)

logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_tokenized(tokenized_dir: Path) -> DatasetDict:
    if not tokenized_dir.exists():
        raise FileNotFoundError(
            f"Tokenized dataset not found at {tokenized_dir}. "
            "Run scripts/tokenize_dataset.py first."
        )
    ds = load_from_disk(str(tokenized_dir))
    if not isinstance(ds, DatasetDict):
        raise TypeError(f"Expected a DatasetDict, got {type(ds).__name__}.")
    for split_name in ("train", "validation"):
        if split_name not in ds:
            raise KeyError(f"Tokenized DatasetDict is missing the {split_name!r} split.")
    return ds


def _maybe_subsample(split: Dataset, n: int | None, seed: int) -> Dataset:
    """Return a deterministic random subset (or the full split if n is None/<=0)."""
    if n is None or n <= 0 or n >= len(split):
        return split
    return split.shuffle(seed=seed).select(range(n))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3: fine-tune CodeBERT for AI-vs-human binary classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tokenized-dir", type=Path, default=config.TOKENIZED_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=config.BASELINE_MODEL_DIR)
    parser.add_argument("--model-name", type=str, default=config.MODEL_NAME)
    parser.add_argument("--num-epochs", type=int, default=config.DEFAULT_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=config.DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=config.DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=config.DEFAULT_WARMUP_RATIO)
    parser.add_argument(
        "--gradient-clip-norm", type=float, default=config.DEFAULT_GRADIENT_CLIP_NORM
    )
    parser.add_argument(
        "--early-stop-patience", type=int, default=config.DEFAULT_EARLY_STOP_PATIENCE
    )
    parser.add_argument("--dropout", type=float, default=config.CLASSIFIER_DROPOUT)
    parser.add_argument("--threshold", type=float, default=config.DECISION_THRESHOLD)
    parser.add_argument(
        "--max-length",
        type=int,
        default=config.MAX_SEQUENCE_LENGTH,
        help=(
            "Effective max_sequence_length used for the run. The tokenized data on disk "
            "already determines the actual cap; this value is recorded in training_config.json "
            "for provenance only."
        ),
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force a torch device (e.g. cpu, cuda, mps). Default: auto-detect.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes. 0 is safest on Windows; >0 only on Linux/macOS.",
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=0,
        help="If >0, randomly subsample the training split to this size (used for smoke tests).",
    )
    parser.add_argument(
        "--limit-val",
        type=int,
        default=0,
        help="If >0, randomly subsample the validation split to this size.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)

    logger.info("Loading tokenized dataset from %s", args.tokenized_dir)
    ds = _load_tokenized(args.tokenized_dir)

    train_split = _maybe_subsample(ds["train"], args.limit_train, args.seed)
    val_split = _maybe_subsample(ds["validation"], args.limit_val, args.seed)
    logger.info("Train size: %d | Val size: %d", len(train_split), len(val_split))

    # Reuse the tokenizer that produced the cached arrays so the special-token
    # vocabulary lines up exactly with the input_ids on disk.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=str(config.HF_CACHE_DIR),
        use_fast=True,
    )

    train_dataset = CodeClassificationDataset(train_split)
    val_dataset = CodeClassificationDataset(val_split)

    train_loader = build_dataloader(
        train_dataset,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = build_dataloader(
        val_dataset,
        tokenizer,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = select_device(args.device)

    model = CodeBertBinaryClassifier(
        model_name=args.model_name,
        dropout_prob=args.dropout,
    )
    logger.info(
        "Model parameters: %s (trainable: %s)",
        f"{model.num_parameters():,}",
        f"{model.num_parameters(trainable_only=True):,}",
    )

    training_config = TrainingConfig(
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_clip_norm=args.gradient_clip_norm,
        max_sequence_length=args.max_length,
        classifier_dropout=args.dropout,
        decision_threshold=args.threshold,
        early_stop_patience=args.early_stop_patience,
        random_seed=args.seed,
    )

    result = train(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=training_config,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
    )

    logger.info(
        "Training finished. Best epoch=%d (val_loss=%.4f) | checkpoint=%s",
        result.best_epoch,
        result.best_val_loss,
        result.checkpoint_dir,
    )
    if result.best_val_metrics is not None:
        logger.info("Best validation metrics:\n%s", result.best_val_metrics.pretty())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
