"""Language filtering and basic cleaning.

The project proposal scopes the analysis to Python only, so we drop samples
from any other language *after* loading the full dataset. We also strip out
rows with empty or obviously degenerate code bodies -- those contribute no
signal and would skew tokenizer length statistics.
"""

from __future__ import annotations

import logging

import pandas as pd

from ai_code_detector import config

logger = logging.getLogger(__name__)

# Minimum number of non-whitespace characters for a sample to be retained.
# A three-character threshold rejects empty strings, single-line throwaways,
# and stray punctuation without discarding legitimate short snippets.
_MIN_CODE_LENGTH: int = 3


def filter_to_python(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose language is (an alias of) Python.

    Args:
        df: Canonical-schema DataFrame (from
            :func:`ai_code_detector.data.loading.normalize_schema`).

    Returns:
        Filtered DataFrame containing Python samples only.
    """
    if config.LANGUAGE_COLUMN not in df.columns:
        raise KeyError(
            f"DataFrame is missing the canonical {config.LANGUAGE_COLUMN!r} column; "
            "pass a normalized DataFrame."
        )

    mask = df[config.LANGUAGE_COLUMN].isin(config.PYTHON_LANGUAGE_ALIASES)
    kept = df.loc[mask].copy()
    logger.info(
        "Language filter: kept %d/%d rows (%.1f%%) matching %s",
        len(kept),
        len(df),
        100.0 * len(kept) / max(len(df), 1),
        sorted(config.PYTHON_LANGUAGE_ALIASES),
    )
    return kept


def drop_empty_and_duplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop samples with empty code bodies or duplicate code strings.

    De-duplication uses the normalized code string as the key; if the same
    snippet appears with different labels we keep the first occurrence and
    log a warning, since label ambiguity would pollute training.

    Args:
        df: Canonical-schema DataFrame.

    Returns:
        Cleaned DataFrame, reset index.
    """
    before = len(df)

    # Reject null / effectively-empty code.
    non_empty_mask = df[config.CODE_COLUMN].notna() & (
        df[config.CODE_COLUMN].str.strip().str.len() >= _MIN_CODE_LENGTH
    )
    cleaned = df.loc[non_empty_mask].copy()
    dropped_empty = before - len(cleaned)
    if dropped_empty:
        logger.info(
            "Dropped %d empty/short code rows (< %d chars).", dropped_empty, _MIN_CODE_LENGTH
        )

    # Detect duplicate code bodies with conflicting labels -- these are data
    # quality issues worth surfacing but not fatal.
    dup_mask = cleaned.duplicated(subset=[config.CODE_COLUMN], keep=False)
    if dup_mask.any():
        conflicting = (
            cleaned.loc[dup_mask].groupby(config.CODE_COLUMN)[config.LABEL_COLUMN].nunique()
        )
        num_conflicting = int((conflicting > 1).sum())
        if num_conflicting:
            logger.warning(
                "Found %d code snippets with conflicting labels across duplicates; "
                "keeping first occurrence of each snippet.",
                num_conflicting,
            )

    deduped = cleaned.drop_duplicates(subset=[config.CODE_COLUMN], keep="first").reset_index(
        drop=True
    )
    dropped_dupes = len(cleaned) - len(deduped)
    if dropped_dupes:
        logger.info("Dropped %d duplicate code rows.", dropped_dupes)

    logger.info("After cleaning: %d rows (from %d).", len(deduped), before)
    return deduped
