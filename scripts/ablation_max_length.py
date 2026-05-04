"""Phase 5 ablation: how does ``max_sequence_length`` affect val metrics?

For each value in ``--lengths`` (default 128, 256, 512) we:

1. Re-tokenize the existing splits at the given ``max_length`` (truncation only).
2. Fine-tune a fresh CodeBERT classifier under
   ``models/ablation_maxlen/L<length>/`` (default hyperparameters).
3. Record the validation metrics in
   ``reports/metrics/ablation_max_length.csv``.

This is a single-axis ablation -- all other hyperparameters are held at their
configured defaults so the table is interpretable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
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
from ai_code_detector.training.loop import select_device, set_seed, train  # noqa: E402

logger = logging.getLogger("ablation_max_length")


def _retruncate(ds: DatasetDict, max_length: int) -> DatasetDict:
    """Re-truncate already-tokenized inputs to ``max_length``.

    The Phase 2 tokenization caps sequences at :data:`config.MAX_SEQUENCE_LENGTH`
    (512). Slicing the token / mask lists is functionally identical to
    re-tokenizing with a smaller ``max_length`` *for truncation only*, and is
    much cheaper.
    """

    def slice_batch(batch: dict) -> dict:
        return {
            "input_ids": [ids[:max_length] for ids in batch["input_ids"]],
            "attention_mask": [mask[:max_length] for mask in batch["attention_mask"]],
            "label": batch["label"],
        }

    return DatasetDict(
        {
            name: split.map(slice_batch, batched=True, desc=f"truncate {name} -> {max_length}")
            for name, split in ds.items()
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 ablation over max_sequence_length.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tokenized-dir", type=Path, default=config.TOKENIZED_DIR)
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=config.MODELS_DIR / "ablation_max_length",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=config.METRICS_DIR / "ablation_max_length.csv",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512],
        help="Max sequence lengths to ablate over.",
    )
    parser.add_argument("--num-epochs", type=int, default=config.DEFAULT_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)

    base: DatasetDict = load_from_disk(str(args.tokenized_dir))
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        cache_dir=str(config.HF_CACHE_DIR),
        use_fast=True,
    )
    device = select_device(args.device)

    rows: list[dict] = []
    for max_length in args.lengths:
        if max_length > config.MAX_SEQUENCE_LENGTH:
            logger.warning(
                "Skipping max_length=%d (> base tokenization cap %d).",
                max_length,
                config.MAX_SEQUENCE_LENGTH,
            )
            continue

        logger.info("=== Ablation: max_length=%d ===", max_length)
        ds = _retruncate(base, max_length)

        train_ds: Dataset = ds["train"]
        val_ds: Dataset = ds["validation"]

        train_loader = build_dataloader(
            CodeClassificationDataset(train_ds),
            tokenizer,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = build_dataloader(
            CodeClassificationDataset(val_ds),
            tokenizer,
            batch_size=config.DEFAULT_EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=args.num_workers,
        )

        model = CodeBertBinaryClassifier()
        cfg = TrainingConfig(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_sequence_length=max_length,
            random_seed=args.seed,
        )
        run_dir = args.ablation_root / f"L{max_length}"
        result = train(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            training_config=cfg,
            checkpoint_dir=run_dir,
            device=device,
        )

        if result.best_val_metrics is None:
            logger.warning("No best metrics for max_length=%d", max_length)
            continue

        rows.append(
            {
                "max_length": max_length,
                "best_epoch": result.best_epoch,
                "val_loss": result.best_val_loss,
                "val_accuracy": result.best_val_metrics.accuracy,
                "val_macro_f1": result.best_val_metrics.macro_f1,
                "val_auc_roc": result.best_val_metrics.auc_roc,
                "checkpoint_dir": str(run_dir),
            }
        )

    if not rows:
        logger.warning("No ablation results to report.")
        return 1

    results_df = pd.DataFrame(rows).sort_values("max_length")
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.results_path, index=False)
    logger.info(
        "Wrote ablation table -> %s\n%s", args.results_path, results_df.to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
