"""Phase 6 ablation: how much does the model rely on comments / whitespace?

For each transform in
:data:`ai_code_detector.features.code_normalization.TRANSFORMS`, we:

1. Apply the transform to every Python sample (keeps the same labels and splits).
2. Re-tokenize with the CodeBERT tokenizer.
3. Fine-tune a fresh CodeBERT classifier under
   ``models/ablation_input_format/<transform-name>/``.
4. Record the validation metrics in
   ``reports/metrics/ablation_input_format.csv``.

A drop in metrics under ``no_comments`` or ``minified`` indicates the model
was leaning on stylistic surface features (comment density, whitespace
patterns) rather than purely structural ones.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ai_code_detector import config  # noqa: E402
from ai_code_detector.data.torch_dataset import (  # noqa: E402
    CodeClassificationDataset,
    build_dataloader,
)
from ai_code_detector.features.code_normalization import TRANSFORMS  # noqa: E402
from ai_code_detector.features.tokenization import load_tokenizer  # noqa: E402
from ai_code_detector.logging_utils import configure_logging  # noqa: E402
from ai_code_detector.models.classifier import CodeBertBinaryClassifier  # noqa: E402
from ai_code_detector.training.checkpoint import TrainingConfig  # noqa: E402
from ai_code_detector.training.loop import select_device, set_seed, train  # noqa: E402

logger = logging.getLogger("ablation_input_format")


def _build_tokenized_split(
    split_df: pd.DataFrame,
    transform_name: str,
    tokenizer,
    max_length: int,
) -> Dataset:
    """Apply a transform to ``split_df`` and tokenize, returning an HF Dataset."""
    transform = TRANSFORMS[transform_name]
    transformed_codes = [transform(code) for code in split_df[config.CODE_COLUMN].tolist()]

    enc = tokenizer(
        transformed_codes,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    return Dataset.from_dict(
        {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": split_df[config.LABEL_COLUMN].astype(int).tolist(),
        }
    )


def _build_tokenized_dict(
    split_dir: Path,
    transform_name: str,
    tokenizer,
    max_length: int,
) -> DatasetDict:
    """Materialize a transform-specific :class:`DatasetDict`."""
    out: dict[str, Dataset] = {}
    for split_name in config.SPLIT_NAMES:
        path = split_dir / f"{split_name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        df = pd.read_parquet(path)
        out[split_name] = _build_tokenized_split(df, transform_name, tokenizer, max_length)
    return DatasetDict(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6 ablation over input-format transforms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split-dir", type=Path, default=config.SPLIT_DIR)
    parser.add_argument(
        "--ablation-root",
        type=Path,
        default=config.MODELS_DIR / "ablation_input_format",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=config.METRICS_DIR / "ablation_input_format.csv",
    )
    parser.add_argument(
        "--transforms",
        type=str,
        nargs="+",
        default=list(TRANSFORMS.keys()),
        help="Subset of TRANSFORMS to ablate. Default: all.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Tokenizer truncation length. Lower is faster on CPU.",
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

    unknown = [name for name in args.transforms if name not in TRANSFORMS]
    if unknown:
        raise SystemExit(f"Unknown transforms {unknown!r}; available: {sorted(TRANSFORMS)}")

    tokenizer = load_tokenizer(config.MODEL_NAME)
    device = select_device(args.device)

    rows: list[dict] = []
    for transform_name in args.transforms:
        logger.info("=== Ablation: input-format transform = %s ===", transform_name)
        ds = _build_tokenized_dict(args.split_dir, transform_name, tokenizer, args.max_length)

        train_loader = build_dataloader(
            CodeClassificationDataset(ds["train"]),
            tokenizer,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = build_dataloader(
            CodeClassificationDataset(ds["validation"]),
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
            max_sequence_length=args.max_length,
            random_seed=args.seed,
            extra={"input_format_transform": transform_name},
        )
        run_dir = args.ablation_root / transform_name
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
            logger.warning("No best metrics for transform=%s", transform_name)
            continue

        rows.append(
            {
                "transform": transform_name,
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

    results_df = pd.DataFrame(rows)
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.results_path, index=False)
    logger.info(
        "Wrote ablation table -> %s\n%s",
        args.results_path,
        results_df.to_string(index=False),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
