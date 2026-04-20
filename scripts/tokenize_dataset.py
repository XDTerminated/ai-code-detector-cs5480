"""Phase 2 orchestrator: turn the split Parquet files into tokenized Arrow datasets.

Reads the train / validation / test Parquet files produced by
``prepare_dataset.py``, runs them through the CodeBERT fast tokenizer, and
saves the result as a HuggingFace :class:`~datasets.DatasetDict` on disk.

Run with::

    uv run python scripts/tokenize_dataset.py

The tokenized artifact is directly loadable with
``datasets.load_from_disk(config.TOKENIZED_DIR)`` and consumable by the
training loop built in Phase 3.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ai_code_detector import config  # noqa: E402
from ai_code_detector.data.splitting import DatasetSplits  # noqa: E402
from ai_code_detector.features.tokenization import (  # noqa: E402
    load_tokenizer,
    save_tokenized,
    summarize_token_lengths,
    tokenize_splits,
)
from ai_code_detector.logging_utils import configure_logging  # noqa: E402

logger = logging.getLogger("tokenize_dataset")


def _load_splits_from_parquet(split_dir: Path) -> DatasetSplits:
    """Materialize a :class:`DatasetSplits` from the Parquet files on disk."""
    frames: dict[str, pd.DataFrame] = {}
    for split_name in config.SPLIT_NAMES:
        path = split_dir / f"{split_name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing split file: {path}. Run scripts/prepare_dataset.py first."
            )
        frames[split_name] = pd.read_parquet(path)
        logger.info("Loaded %s: %d rows from %s", split_name, len(frames[split_name]), path)
    return DatasetSplits(
        train=frames["train"],
        validation=frames["validation"],
        test=frames["test"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: tokenize the Phase 1 splits with the CodeBERT tokenizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=config.SPLIT_DIR,
        help="Directory containing train/validation/test Parquet files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.TOKENIZED_DIR,
        help="Destination directory for the tokenized HuggingFace DatasetDict.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=config.MODEL_NAME,
        help="HuggingFace model whose tokenizer should be used.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=config.MAX_SEQUENCE_LENGTH,
        help="Truncation length in tokens.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of rows per tokenization batch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()

    logger.info("Starting Phase 2 tokenization")
    splits = _load_splits_from_parquet(args.split_dir)

    tokenizer = load_tokenizer(args.model_name)

    tokenized = tokenize_splits(
        splits,
        tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    logger.info(
        "Sequence-length summary (post-tokenization):\n%s",
        summarize_token_lengths(tokenized).to_string(float_format="%.1f"),
    )

    save_tokenized(tokenized, args.out_dir)
    logger.info("Phase 2 complete. Tokenized dataset at %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
