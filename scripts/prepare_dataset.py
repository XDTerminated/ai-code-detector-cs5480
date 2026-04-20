"""Phase 1 orchestrator: raw -> canonical -> Python-only -> stratified splits.

Pipeline stages
---------------
1. **Load** every supported tabular file under ``data/raw/``.
2. **Normalize** the schema onto the canonical ``(code, label, language)``
   triple, coping with both long-form and paired-form inputs.
3. **Filter** to Python only, drop empty snippets and duplicates.
4. **Split** into train / validation / test with a stratified, seeded split.
5. **Persist** each stage so downstream tools (EDA, tokenization, training)
   can consume a clean on-disk artifact without re-running the whole chain.

Run with::

    uv run python scripts/prepare_dataset.py

All paths and ratios come from :mod:`ai_code_detector.config`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Make ``src/`` importable when this script is run directly (``python scripts/...``).
# If the package is installed (e.g., via ``uv sync``) this is a no-op.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ai_code_detector import config  # noqa: E402
from ai_code_detector.data.filtering import drop_empty_and_duplicate, filter_to_python  # noqa: E402
from ai_code_detector.data.loading import load_raw_dataset, normalize_schema  # noqa: E402
from ai_code_detector.data.splitting import DatasetSplits, stratified_split  # noqa: E402
from ai_code_detector.logging_utils import configure_logging  # noqa: E402

logger = logging.getLogger("prepare_dataset")


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def _report_class_balance(df: pd.DataFrame, *, label: str) -> None:
    """Log class distribution for a given DataFrame."""
    counts = df[config.LABEL_COLUMN].value_counts().sort_index()
    total = int(counts.sum())
    pieces = []
    for lbl, count in counts.items():
        name = config.LABEL_NAMES.get(int(lbl), str(lbl))
        pieces.append(f"{name}={count} ({100.0 * count / total:.1f}%)")
    logger.info("%s class balance (n=%d): %s", label, total, ", ".join(pieces))


def _report_splits(splits: DatasetSplits) -> None:
    """Log sizes and class balance for each split."""
    for split_name, split_df in splits.as_dict().items():
        _report_class_balance(split_df, label=f"[{split_name}]")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def prepare_dataset(
    *,
    raw_dir: Path,
    interim_path: Path,
    split_dir: Path,
) -> DatasetSplits:
    """Run the full Phase 1 pipeline and persist intermediate artifacts.

    Args:
        raw_dir: Directory containing the extracted HumanVSAI_CodeDataset.
        interim_path: Parquet path for the filtered Python-only dataset.
        split_dir: Directory for the three split Parquet files.

    Returns:
        The resulting :class:`DatasetSplits`.
    """
    # 1. Load + normalize.
    logger.info("Stage 1/4: loading raw dataset from %s", raw_dir)
    raw = load_raw_dataset(raw_dir)
    logger.info("Stage 1/4: normalizing schema")
    canonical = normalize_schema(raw)
    _report_class_balance(canonical, label="[raw-all-languages]")
    logger.info(
        "Languages found in raw data: %s",
        sorted(canonical[config.LANGUAGE_COLUMN].unique().tolist()),
    )

    # 2. Filter to Python.
    logger.info("Stage 2/4: filtering to Python-only")
    python_df = filter_to_python(canonical)
    python_df = drop_empty_and_duplicate(python_df)
    _report_class_balance(python_df, label="[python-only]")

    interim_path.parent.mkdir(parents=True, exist_ok=True)
    python_df.to_parquet(interim_path, index=False)
    logger.info("Wrote %d Python rows to %s", len(python_df), interim_path)

    # 3. Stratified split.
    logger.info("Stage 3/4: stratified train/val/test split")
    splits = stratified_split(python_df)
    _report_splits(splits)

    # 4. Persist splits.
    logger.info("Stage 4/4: writing splits to %s", split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in splits.as_dict().items():
        out = split_dir / f"{split_name}.parquet"
        split_df.to_parquet(out, index=False)
        logger.info("  %s -> %s (%d rows)", split_name, out, len(split_df))

    return splits


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: load, normalize, filter to Python, and split the HumanVSAI_CodeDataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=config.RAW_DATA_DIR,
        help="Directory containing the extracted raw dataset.",
    )
    parser.add_argument(
        "--interim-path",
        type=Path,
        default=config.PYTHON_ONLY_PARQUET,
        help="Output Parquet path for the filtered Python-only dataset.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=config.SPLIT_DIR,
        help="Output directory for the train/validation/test Parquet files.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Emit DEBUG-level logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    logger.info("Starting Phase 1 dataset preparation")
    try:
        prepare_dataset(
            raw_dir=args.raw_dir,
            interim_path=args.interim_path,
            split_dir=args.split_dir,
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except (KeyError, ValueError) as exc:
        logger.error("Pipeline failed: %s", exc)
        return 2

    logger.info("Phase 1 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
