"""Phase 1 EDA: characterize the Python-only subset before modeling.

Computes and saves:

* Per-class counts (overall and per split).
* Code-length distributions (characters, lines) per class.
* Duplicate and near-duplicate statistics.
* Basic token-length proxy (whitespace tokens) so we can sanity-check
  :data:`ai_code_detector.config.MAX_SEQUENCE_LENGTH` before paying for
  real tokenization.

Outputs go to ``reports/figures/`` (PNG plots) and the console (tables).

Run with::

    uv run python scripts/analyze_dataset.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend -- safe for headless runs.

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ai_code_detector import config  # noqa: E402
from ai_code_detector.logging_utils import configure_logging  # noqa: E402

logger = logging.getLogger("analyze_dataset")

sns.set_theme(style="whitegrid", context="talk")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_python_dataset(interim_path: Path, split_dir: Path) -> pd.DataFrame:
    """Load the Python-only dataset plus per-row split labels for grouping."""
    if not interim_path.exists():
        raise FileNotFoundError(
            f"Interim Python-only dataset not found: {interim_path}. "
            "Run scripts/prepare_dataset.py first."
        )
    df = pd.read_parquet(interim_path)

    # Tag each row with its split so we can facet plots by split. Rows that
    # somehow fall outside the splits (shouldn't happen) are marked 'unknown'.
    split_col = pd.Series(["unknown"] * len(df), index=df.index, dtype="object")
    for split_name in config.SPLIT_NAMES:
        split_path = split_dir / f"{split_name}.parquet"
        if split_path.exists():
            split_rows = pd.read_parquet(split_path, columns=[config.CODE_COLUMN])
            # Match by code string: splits were created from the same DataFrame so
            # exact string equality is a safe key (after de-duplication in Phase 1).
            split_col.loc[df[config.CODE_COLUMN].isin(split_rows[config.CODE_COLUMN])] = split_name
    df = df.assign(split=split_col.values)
    return df


# ---------------------------------------------------------------------------
# Feature engineering for EDA
# ---------------------------------------------------------------------------
def _augment_with_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add character-count, line-count, and whitespace-token-count columns."""
    out = df.copy()
    out["n_chars"] = out[config.CODE_COLUMN].str.len()
    out["n_lines"] = out[config.CODE_COLUMN].str.count("\n") + 1
    # Cheap tokenization proxy -- real CodeBERT tokenization happens in Phase 2.
    out["n_whitespace_tokens"] = out[config.CODE_COLUMN].str.split().map(len)
    out["label_name"] = out[config.LABEL_COLUMN].map(config.LABEL_NAMES)
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _log_class_balance(df: pd.DataFrame) -> None:
    logger.info("Total rows: %d", len(df))
    overall = (
        df.groupby("label_name")
        .size()
        .rename("count")
        .to_frame()
        .assign(pct=lambda x: 100.0 * x["count"] / x["count"].sum())
    )
    logger.info("Overall class balance:\n%s", overall.to_string(float_format="%.2f"))

    by_split = (
        df.groupby(["split", "label_name"])
        .size()
        .unstack(fill_value=0)
        .assign(total=lambda x: x.sum(axis=1))
    )
    logger.info("Class balance per split:\n%s", by_split.to_string())


def _log_length_stats(df: pd.DataFrame) -> None:
    for column in ("n_chars", "n_lines", "n_whitespace_tokens"):
        summary = df.groupby("label_name")[column].describe(percentiles=[0.5, 0.9, 0.95, 0.99])
        logger.info("%s distribution by class:\n%s", column, summary.to_string(float_format="%.1f"))


def _log_duplicates(df: pd.DataFrame) -> None:
    n_dupes = int(df.duplicated(subset=[config.CODE_COLUMN]).sum())
    logger.info("Exact-duplicate code rows remaining: %d", n_dupes)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_class_balance(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df.groupby("label_name").size().reset_index(name="count")
    sns.barplot(data=counts, x="label_name", y="count", ax=ax, hue="label_name", legend=False)
    ax.set_title("Class balance (Python-only)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Samples")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)
    fig.tight_layout()
    _save(fig, out_dir / "class_balance.png")


def _plot_length_distribution(df: pd.DataFrame, column: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    # Clip at 99th percentile so the tail doesn't dominate the axis.
    p99 = float(df[column].quantile(0.99))
    clipped = df[df[column] <= p99]
    sns.histplot(
        data=clipped,
        x=column,
        hue="label_name",
        bins=60,
        stat="density",
        common_norm=False,
        element="step",
        ax=ax,
    )
    ax.set_title(f"{column} distribution by class (clipped at p99={p99:.0f})")
    ax.set_xlabel(column)
    ax.set_ylabel("Density")
    fig.tight_layout()
    _save(fig, out_dir / f"distribution_{column}.png")


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 EDA: class balance, length distributions, duplicates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--interim-path",
        type=Path,
        default=config.PYTHON_ONLY_PARQUET,
        help="Parquet file produced by prepare_dataset.py.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=config.SPLIT_DIR,
        help="Directory containing train/validation/test Parquet files.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=config.FIGURES_DIR,
        help="Directory to write PNG plots to.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()

    logger.info("Loading Python-only dataset")
    df = _load_python_dataset(args.interim_path, args.split_dir)
    df = _augment_with_length_features(df)

    logger.info("--- Tables ---")
    _log_class_balance(df)
    _log_length_stats(df)
    _log_duplicates(df)

    logger.info("--- Plots ---")
    _plot_class_balance(df, args.figures_dir)
    for column in ("n_chars", "n_lines", "n_whitespace_tokens"):
        _plot_length_distribution(df, column, args.figures_dir)

    logger.info("EDA complete. Figures written to %s", args.figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
