"""Phase 4: small grid search over learning rate, batch size, and epochs.

For each configuration we:

1. Fine-tune CodeBERT into ``models/sweep/<config-id>/``.
2. Evaluate on the validation split.
3. Record the validation metrics in a single ``sweep_results.csv`` table.

The best configuration (by val macro-F1) is reported at the end. Re-run with
the chosen config -- not from inside this script -- so the final retrained
checkpoint lives in a clean directory like ``models/best/``.

Note on cost
------------
Each configuration fine-tunes a 125M-parameter model. On CPU this is ~30
minutes per run, so this script is opt-in: pass ``--configs`` to limit the
sweep to a small subset. Defaults are the canonical RoBERTa-fine-tuning grid
from the literature.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pandas as pd
from datasets import DatasetDict, load_from_disk

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from ai_code_detector import config  # noqa: E402
from ai_code_detector.data.torch_dataset import (  # noqa: E402
    CodeClassificationDataset,
    build_dataloader,
)
from ai_code_detector.evaluation.predict import predict  # noqa: E402
from ai_code_detector.logging_utils import configure_logging  # noqa: E402
from ai_code_detector.models.classifier import CodeBertBinaryClassifier  # noqa: E402
from ai_code_detector.training.checkpoint import TrainingConfig  # noqa: E402
from ai_code_detector.training.loop import (  # noqa: E402
    select_device,
    set_seed,
    train,
)
from ai_code_detector.training.metrics import compute_classification_metrics  # noqa: E402

logger = logging.getLogger("hyperparameter_search")


@dataclass(frozen=True, slots=True)
class SweepCell:
    learning_rate: float
    batch_size: int
    num_epochs: int

    @property
    def label(self) -> str:
        return f"lr{self.learning_rate:.0e}_bs{self.batch_size}_ep{self.num_epochs}"


def _default_grid() -> list[SweepCell]:
    """Standard small grid for transformer fine-tuning."""
    learning_rates = (1e-5, 2e-5, 5e-5)
    batch_sizes = (16,)
    epoch_counts = (3,)
    return [
        SweepCell(learning_rate=lr, batch_size=bs, num_epochs=ne)
        for lr, bs, ne in product(learning_rates, batch_sizes, epoch_counts)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4: hyperparameter sweep over LR / batch / epochs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tokenized-dir", type=Path, default=config.TOKENIZED_DIR)
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=config.MODELS_DIR / "sweep",
        help="Each config's checkpoint is written under <sweep-root>/<config-label>/.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=config.METRICS_DIR / "sweep_results.csv",
        help="Where to write the aggregated sweep table.",
    )
    parser.add_argument(
        "--configs",
        type=int,
        default=0,
        help="If >0, take only the first N configs from the grid. Useful for smoke-testing.",
    )
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    set_seed(args.seed)

    grid = _default_grid()
    if args.configs and args.configs > 0:
        grid = grid[: args.configs]
    logger.info("Sweep grid (%d configs): %s", len(grid), [c.label for c in grid])

    ds: DatasetDict = load_from_disk(str(args.tokenized_dir))
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        cache_dir=str(config.HF_CACHE_DIR),
        use_fast=True,
    )
    val_dataset = CodeClassificationDataset(ds["validation"])
    device = select_device(args.device)

    rows: list[dict] = []
    for cell in grid:
        run_dir = args.sweep_root / cell.label
        logger.info("=== Sweep cell: %s ===", cell.label)

        train_loader = build_dataloader(
            CodeClassificationDataset(ds["train"]),
            tokenizer,
            batch_size=cell.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        val_loader = build_dataloader(
            val_dataset,
            tokenizer,
            batch_size=config.DEFAULT_EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=args.num_workers,
        )

        model = CodeBertBinaryClassifier()
        cfg = TrainingConfig(
            num_epochs=cell.num_epochs,
            batch_size=cell.batch_size,
            learning_rate=cell.learning_rate,
            random_seed=args.seed,
        )
        result = train(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            training_config=cfg,
            checkpoint_dir=run_dir,
            device=device,
        )

        # Re-run inference on val with the best checkpoint loaded into the
        # in-memory model -- ``train`` leaves the model in last-epoch state,
        # so we use the recorded best metrics instead of recomputing.
        best_metrics = result.best_val_metrics
        if best_metrics is None:
            logger.warning("No best metrics recorded for %s; skipping row.", cell.label)
            continue

        rows.append(
            {
                "config": cell.label,
                "learning_rate": cell.learning_rate,
                "batch_size": cell.batch_size,
                "num_epochs": cell.num_epochs,
                "best_epoch": result.best_epoch,
                "val_loss": result.best_val_loss,
                "val_accuracy": best_metrics.accuracy,
                "val_macro_f1": best_metrics.macro_f1,
                "val_auc_roc": best_metrics.auc_roc,
                "checkpoint_dir": str(run_dir),
            }
        )

    if not rows:
        logger.warning("No sweep results to report.")
        return 1

    results_df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.results_path, index=False)
    logger.info(
        "Wrote sweep results -> %s\n%s", args.results_path, results_df.to_string(index=False)
    )

    best = results_df.iloc[0].to_dict()
    logger.info("Best by val_macro_f1:\n%s", json.dumps(best, indent=2, default=str))

    # Convenience: also evaluate the best checkpoint on val once more (sanity).
    best_dir = Path(str(best["checkpoint_dir"]))
    logger.info("Re-loading best checkpoint at %s for final sanity-check.", best_dir)
    from ai_code_detector.training.checkpoint import load_checkpoint

    model, tokenizer, _cfg = load_checkpoint(best_dir)
    final_loader = build_dataloader(
        val_dataset,
        tokenizer,
        batch_size=config.DEFAULT_EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
    )
    final = predict(model, final_loader, device, threshold=config.DECISION_THRESHOLD)
    sanity = compute_classification_metrics(
        y_true=final.labels,
        y_proba=final.probabilities,
    )
    logger.info("Sanity-check val metrics on best checkpoint:\n%s", sanity.pretty())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
