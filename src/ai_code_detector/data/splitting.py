"""Stratified train / validation / test splitting.

We split in *two* sklearn calls rather than one, because ``train_test_split``
only supports a single split at a time. The two-step procedure preserves class
proportions across all three splits to within one sample, which is exactly
what stratification promises for a single call.

The split is deterministic given :data:`ai_code_detector.config.RANDOM_SEED`,
so reruns on the same data produce byte-identical outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from ai_code_detector import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """Immutable container for the three split DataFrames."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        """Return the splits keyed by the canonical split names."""
        return {"train": self.train, "validation": self.validation, "test": self.test}

    def sizes(self) -> dict[str, int]:
        """Return the row count of each split."""
        return {name: len(df) for name, df in self.as_dict().items()}


def _validate_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    """Confirm the three ratios are positive and sum to 1.0 (within tolerance)."""
    if min(train_ratio, validation_ratio, test_ratio) <= 0.0:
        raise ValueError(
            f"Split ratios must all be positive; got "
            f"train={train_ratio}, val={validation_ratio}, test={test_ratio}."
        )
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0; got {total:.6f}.")


def stratified_split(
    df: pd.DataFrame,
    *,
    train_ratio: float = config.TRAIN_RATIO,
    validation_ratio: float = config.VALIDATION_RATIO,
    test_ratio: float = config.TEST_RATIO,
    random_seed: int = config.RANDOM_SEED,
    stratify_column: str = config.LABEL_COLUMN,
) -> DatasetSplits:
    """Split ``df`` into train / validation / test, stratified by label.

    Two-step procedure:

    1. Carve off the test set (``test_ratio`` of rows).
    2. Split the remaining rows into train and validation such that the
       validation set is exactly ``validation_ratio`` of the *original* data.

    Args:
        df: Input DataFrame (typically the cleaned Python-only subset).
        train_ratio: Proportion of samples for training.
        validation_ratio: Proportion of samples for validation.
        test_ratio: Proportion of samples for testing.
        random_seed: Seed for reproducibility.
        stratify_column: Column to stratify on. Defaults to the label column.

    Returns:
        A :class:`DatasetSplits` with the three splits.
    """
    _validate_ratios(train_ratio, validation_ratio, test_ratio)

    if stratify_column not in df.columns:
        raise KeyError(
            f"Cannot stratify on {stratify_column!r}; available columns: {list(df.columns)}."
        )

    n_total = len(df)
    if n_total < 3:
        raise ValueError(f"Need at least 3 samples to split, got {n_total}.")

    stratify_target = df[stratify_column]

    # Step 1: peel off the test set.
    train_plus_val, test = train_test_split(
        df,
        test_size=test_ratio,
        stratify=stratify_target,
        random_state=random_seed,
        shuffle=True,
    )

    # Step 2: split the remainder into train and validation. The fraction
    # passed to ``test_size`` is expressed relative to ``train_plus_val``, so
    # we rescale from the original ratio.
    val_fraction_of_remainder = validation_ratio / (train_ratio + validation_ratio)
    train, validation = train_test_split(
        train_plus_val,
        test_size=val_fraction_of_remainder,
        stratify=train_plus_val[stratify_column],
        random_state=random_seed,
        shuffle=True,
    )

    splits = DatasetSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )

    logger.info(
        "Stratified split (seed=%d): train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%).",
        random_seed,
        len(splits.train),
        100.0 * len(splits.train) / n_total,
        len(splits.validation),
        100.0 * len(splits.validation) / n_total,
        len(splits.test),
        100.0 * len(splits.test) / n_total,
    )
    return splits
