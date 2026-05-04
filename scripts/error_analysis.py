"""Phase 5: dump misclassified samples and basic error statistics.

For each misclassified test sample we record:

* the original Python code (pulled from the Phase 1 split Parquet),
* the true class and the predicted class,
* the model's predicted probability of class 1 (AI),
* and a few length features so we can see whether errors cluster on long /
  short snippets.

Outputs:

* ``reports/metrics/<run-name>_errors.csv`` -- one row per misclassified test sample.
* ``reports/metrics/<run-name>_error_summary.json`` -- aggregate stats:
  total errors, error rate per class, mean/median probability on errors,
  length distribution comparison vs. correctly-classified samples.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_from_disk

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ai_code_detector import config  # noqa: E402
from ai_code_detector.data.torch_dataset import (  # noqa: E402
    CodeClassificationDataset,
    build_dataloader,
)
from ai_code_detector.evaluation.predict import predict  # noqa: E402
from ai_code_detector.logging_utils import configure_logging  # noqa: E402
from ai_code_detector.training.checkpoint import load_checkpoint  # noqa: E402
from ai_code_detector.training.loop import select_device  # noqa: E402

logger = logging.getLogger("error_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5: misclassified-sample dump and error statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=config.BASELINE_MODEL_DIR)
    parser.add_argument("--tokenized-dir", type=Path, default=config.TOKENIZED_DIR)
    parser.add_argument(
        "--split-parquet",
        type=Path,
        default=config.SPLIT_DIR / "test.parquet",
        help="Parquet file with the original code strings (for human-readable errors).",
    )
    parser.add_argument("--metrics-dir", type=Path, default=config.METRICS_DIR)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Output filename prefix. Defaults to the checkpoint dir name.",
    )
    parser.add_argument("--max-error-rows", type=int, default=200)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()

    run_name = args.run_name or args.checkpoint_dir.name

    logger.info("Loading checkpoint from %s", args.checkpoint_dir)
    model, tokenizer, training_config = load_checkpoint(args.checkpoint_dir)

    logger.info("Loading tokenized test split from %s", args.tokenized_dir)
    ds: DatasetDict = load_from_disk(str(args.tokenized_dir))
    test_split = ds["test"]
    test_dataset = CodeClassificationDataset(test_split)
    test_loader = build_dataloader(
        test_dataset,
        tokenizer,
        batch_size=training_config.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = select_device(args.device)

    logger.info("Predicting on %d test samples", len(test_dataset))
    pred = predict(
        model,
        test_loader,
        device,
        threshold=training_config.decision_threshold,
        desc="error_analysis",
    )

    if not args.split_parquet.exists():
        raise FileNotFoundError(
            f"Split parquet not found: {args.split_parquet}. Run scripts/prepare_dataset.py first."
        )
    test_parquet = pd.read_parquet(args.split_parquet)
    if len(test_parquet) != len(pred.labels):
        raise RuntimeError(
            f"Row count mismatch between tokenized test split ({len(pred.labels)}) "
            f"and parquet test split ({len(test_parquet)}). Did you re-tokenize after "
            "re-splitting?"
        )

    df = pd.DataFrame(
        {
            config.CODE_COLUMN: test_parquet[config.CODE_COLUMN].to_numpy(),
            "label": pred.labels,
            "label_name": [config.LABEL_NAMES[int(label)] for label in pred.labels],
            "prediction": pred.predictions,
            "prediction_name": [
                config.LABEL_NAMES[int(prediction)] for prediction in pred.predictions
            ],
            "prob_ai": pred.probabilities,
            "is_correct": pred.predictions == pred.labels,
        }
    )
    df["n_chars"] = df[config.CODE_COLUMN].str.len()
    df["n_lines"] = df[config.CODE_COLUMN].str.count("\n") + 1

    errors = df.loc[~df["is_correct"]].copy()
    # Sort by confidence -- the model's most-confident wrong predictions are
    # the most diagnostic (they tend to point at labeling issues or systematic
    # blind spots).
    errors["confidence"] = np.where(
        errors["prediction"] == 1,
        errors["prob_ai"],
        1.0 - errors["prob_ai"],
    )
    errors = errors.sort_values("confidence", ascending=False).reset_index(drop=True)

    # Persist errors as CSV for human inspection.
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.metrics_dir / f"{run_name}_errors.csv"
    errors.head(args.max_error_rows).to_csv(csv_path, index=False)
    logger.info(
        "Wrote %d/%d misclassified rows -> %s",
        min(args.max_error_rows, len(errors)),
        len(errors),
        csv_path,
    )

    # Aggregate summary.
    correct = df.loc[df["is_correct"]]
    summary = {
        "n_test": len(df),
        "n_errors": len(errors),
        "error_rate": float(len(errors) / max(len(df), 1)),
        "errors_by_true_class": {
            config.LABEL_NAMES[label]: int((errors["label"] == label).sum()) for label in (0, 1)
        },
        "errors_by_predicted_class": {
            config.LABEL_NAMES[label]: int((errors["prediction"] == label).sum())
            for label in (0, 1)
        },
        "mean_confidence_on_errors": float(errors["confidence"].mean()) if len(errors) else None,
        "median_confidence_on_errors": float(errors["confidence"].median())
        if len(errors)
        else None,
        "length_stats": {
            "errors_n_chars_mean": float(errors["n_chars"].mean()) if len(errors) else None,
            "correct_n_chars_mean": float(correct["n_chars"].mean()) if len(correct) else None,
            "errors_n_lines_mean": float(errors["n_lines"].mean()) if len(errors) else None,
            "correct_n_lines_mean": float(correct["n_lines"].mean()) if len(correct) else None,
        },
    }
    summary_path = args.metrics_dir / f"{run_name}_error_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote error summary -> %s", summary_path)

    logger.info("Error summary:\n%s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
