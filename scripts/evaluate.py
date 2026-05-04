"""Phase 4: evaluate a trained checkpoint on the held-out test split.

Loads the best checkpoint from ``models/<run-name>/`` and dumps:

* ``reports/metrics/<run-name>_test.json`` -- the full metric bundle.
* ``reports/figures/<run-name>_confusion_matrix.png`` -- raw counts.
* ``reports/figures/<run-name>_confusion_matrix_norm.png`` -- row-normalized.
* ``reports/figures/<run-name>_roc_curve.png`` -- ROC curve.
* ``reports/figures/<run-name>_per_class_metrics.png`` -- per-class P/R/F1.
* ``reports/figures/<run-name>_training_curves.png`` -- if a history exists.

Run with::

    uv run python scripts/evaluate.py --checkpoint-dir models/baseline
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from datasets import DatasetDict, load_from_disk

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ai_code_detector import config  # noqa: E402
from ai_code_detector.data.torch_dataset import (  # noqa: E402
    CodeClassificationDataset,
    build_dataloader,
)
from ai_code_detector.evaluation.plots import (  # noqa: E402
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_roc_curve,
    plot_training_curves,
)
from ai_code_detector.evaluation.predict import predict  # noqa: E402
from ai_code_detector.logging_utils import configure_logging  # noqa: E402
from ai_code_detector.training.checkpoint import (  # noqa: E402
    load_checkpoint,
    load_training_history,
)
from ai_code_detector.training.loop import select_device  # noqa: E402
from ai_code_detector.training.metrics import compute_classification_metrics  # noqa: E402

logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4: evaluate a trained checkpoint on the test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=config.BASELINE_MODEL_DIR)
    parser.add_argument("--tokenized-dir", type=Path, default=config.TOKENIZED_DIR)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "validation", "test"),
        help="Which split to evaluate. Test is the canonical reportable split.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Used as the output filename prefix. Defaults to the checkpoint dir name.",
    )
    parser.add_argument("--metrics-dir", type=Path, default=config.METRICS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=config.FIGURES_DIR)
    parser.add_argument("--eval-batch-size", type=int, default=config.DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()

    run_name = args.run_name or args.checkpoint_dir.name

    logger.info("Loading checkpoint from %s", args.checkpoint_dir)
    model, tokenizer, training_config = load_checkpoint(args.checkpoint_dir)

    threshold = args.threshold if args.threshold is not None else training_config.decision_threshold
    logger.info("Using decision threshold = %.3f", threshold)

    logger.info("Loading tokenized dataset from %s", args.tokenized_dir)
    ds: DatasetDict = load_from_disk(str(args.tokenized_dir))
    if args.split not in ds:
        raise KeyError(f"Split {args.split!r} not found; available: {list(ds.keys())}")
    eval_dataset = CodeClassificationDataset(ds[args.split])
    eval_loader = build_dataloader(
        eval_dataset,
        tokenizer,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = select_device(args.device)

    logger.info("Running inference over %s split (%d samples)", args.split, len(eval_dataset))
    pred = predict(model, eval_loader, device, threshold=threshold, desc=f"eval [{args.split}]")
    metrics = compute_classification_metrics(
        y_true=pred.labels,
        y_proba=pred.probabilities,
        threshold=threshold,
    )
    logger.info("Metrics on %s split:\n%s", args.split, metrics.pretty())

    # Persist outputs.
    metrics_path = args.metrics_dir / f"{run_name}_{args.split}.json"
    metrics.to_json(metrics_path)
    logger.info("Wrote metrics JSON -> %s", metrics_path)

    plot_confusion_matrix(
        metrics.confusion_matrix,
        args.figures_dir / f"{run_name}_{args.split}_confusion_matrix.png",
        normalize=False,
        title=f"Confusion matrix ({args.split})",
    )
    plot_confusion_matrix(
        metrics.confusion_matrix,
        args.figures_dir / f"{run_name}_{args.split}_confusion_matrix_norm.png",
        normalize=True,
        title=f"Confusion matrix, row-normalized ({args.split})",
    )
    plot_roc_curve(
        pred.labels,
        pred.probabilities,
        args.figures_dir / f"{run_name}_{args.split}_roc_curve.png",
        auc_value=metrics.auc_roc,
        title=f"ROC curve ({args.split})",
    )
    plot_per_class_metrics(
        metrics.per_class,
        args.figures_dir / f"{run_name}_{args.split}_per_class_metrics.png",
    )

    # Training curves: only meaningful for the training run that produced
    # the checkpoint, not for re-evaluations on a different split.
    history = load_training_history(args.checkpoint_dir)
    if history:
        plot_training_curves(
            history,
            args.figures_dir / f"{run_name}_training_curves.png",
        )
    else:
        logger.info("No training_history.json present; skipping training curves.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
