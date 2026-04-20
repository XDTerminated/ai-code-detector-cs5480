"""CodeBERT tokenization of the split dataframes.

The tokenization stage turns string code snippets into the token-id tensors
that CodeBERT expects. We use the fast (Rust-backed) tokenizer shipped with
the model and materialize a HuggingFace :class:`DatasetDict` for each split so
that downstream training code can consume a standard on-disk artifact.

Design choices
--------------
* **Truncation**: always on, at :data:`ai_code_detector.config.MAX_SEQUENCE_LENGTH`
  (512 tokens -- CodeBERT's native upper bound). Longer snippets are cut from
  the tail rather than dropped.
* **Padding**: *off* at tokenization time. Padding is applied dynamically per
  batch at training time via :class:`~transformers.DataCollatorWithPadding`,
  which is more memory- and compute-efficient than padding everything to 512.
  See :mod:`ai_code_detector.data.torch_dataset` for the collator.
* **Batched ``.map``**: the tokenizer is called in batches, which is two
  orders of magnitude faster than row-wise calls for fast tokenizers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ai_code_detector import config
from ai_code_detector.data.splitting import DatasetSplits

logger = logging.getLogger(__name__)

# Columns that must survive into the tokenized dataset so the training loop
# can build (inputs, labels) batches. Everything else is dropped to keep the
# on-disk artifact small.
_KEPT_COLUMNS: tuple[str, ...] = ("input_ids", "attention_mask", "label")


def load_tokenizer(model_name: str = config.MODEL_NAME) -> PreTrainedTokenizerBase:
    """Load the fast tokenizer for the given HuggingFace model.

    Args:
        model_name: HuggingFace model identifier (default: microsoft/codebert-base).

    Returns:
        A fast tokenizer instance.
    """
    logger.info("Loading tokenizer for %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(config.HF_CACHE_DIR),
        use_fast=True,
    )
    if not tokenizer.is_fast:
        # CodeBERT ships a fast tokenizer; if a slow one slips through it will
        # be ~100x slower on large batches and is worth catching loudly.
        raise RuntimeError(
            f"Expected a fast tokenizer for {model_name!r} but got {type(tokenizer).__name__}."
        )
    logger.info(
        "Tokenizer ready: vocab=%d, model_max_length=%d",
        tokenizer.vocab_size,
        tokenizer.model_max_length,
    )
    return tokenizer


def _splits_to_dataset_dict(splits: DatasetSplits) -> DatasetDict:
    """Wrap a :class:`DatasetSplits` as a HuggingFace :class:`DatasetDict`.

    HuggingFace ``datasets`` is the industry-standard on-disk format for this
    kind of pipeline: Arrow-backed, memory-mapped, and trivially shardable.
    """
    return DatasetDict(
        {
            split_name: Dataset.from_pandas(df, preserve_index=False)
            for split_name, df in splits.as_dict().items()
        }
    )


def tokenize_splits(
    splits: DatasetSplits,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int = config.MAX_SEQUENCE_LENGTH,
    batch_size: int = 1000,
) -> DatasetDict:
    """Tokenize every split and return a :class:`DatasetDict`.

    The output preserves the canonical ``label`` column and adds the tokenizer
    outputs (``input_ids``, ``attention_mask``). Any other source columns are
    removed so the resulting dataset is directly consumable by training code.

    Args:
        splits: The three split DataFrames.
        tokenizer: A fast HuggingFace tokenizer.
        max_length: Truncation length in tokens.
        batch_size: Number of rows per ``.map`` batch.

    Returns:
        Tokenized dataset keyed by split name.
    """
    dataset_dict = _splits_to_dataset_dict(splits)

    def tokenize_batch(batch: dict[str, list]) -> dict[str, list]:
        """Apply the tokenizer to one batch of code strings."""
        return tokenizer(
            batch[config.CODE_COLUMN],
            truncation=True,
            max_length=max_length,
            padding=False,  # Dynamic padding happens at collate time.
            return_attention_mask=True,
        )

    tokenized: dict[str, Dataset] = {}
    for split_name, split_ds in dataset_dict.items():
        logger.info(
            "Tokenizing split %r (%d examples, max_length=%d)...",
            split_name,
            len(split_ds),
            max_length,
        )

        # Columns to drop: everything except the label. We keep the label so
        # it lives alongside the token tensors in the final artifact.
        removable = [c for c in split_ds.column_names if c != config.LABEL_COLUMN]

        tokenized_split = split_ds.map(
            tokenize_batch,
            batched=True,
            batch_size=batch_size,
            remove_columns=removable,
            desc=f"tokenizing {split_name}",
        )

        # Sanity check: every expected column is present.
        missing = set(_KEPT_COLUMNS) - set(tokenized_split.column_names)
        if missing:
            raise RuntimeError(
                f"Tokenized split {split_name!r} is missing columns {sorted(missing)}; "
                f"got {tokenized_split.column_names}."
            )

        tokenized[split_name] = tokenized_split

    return DatasetDict(tokenized)


def save_tokenized(dataset_dict: DatasetDict, out_dir: Path) -> None:
    """Persist a tokenized :class:`DatasetDict` to disk.

    Uses Arrow under the hood; the directory can be re-loaded with
    :func:`datasets.load_from_disk`.

    Args:
        dataset_dict: The tokenized splits.
        out_dir: Destination directory. Created if missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving tokenized dataset to %s", out_dir)
    dataset_dict.save_to_disk(str(out_dir))


def summarize_token_lengths(dataset_dict: DatasetDict) -> pd.DataFrame:
    """Return per-split statistics of post-tokenization sequence length.

    Useful for sanity-checking truncation rate and picking a
    ``max_sequence_length`` for future experiments.

    Args:
        dataset_dict: A tokenized :class:`DatasetDict`.

    Returns:
        DataFrame indexed by split name with columns ``count``, ``mean``,
        ``min``, ``p50``, ``p95``, ``p99``, ``max``, and ``truncated_pct``.
    """
    rows = []
    for split_name, split_ds in dataset_dict.items():
        lengths = pd.Series([len(ids) for ids in split_ds["input_ids"]], dtype="int64")
        rows.append(
            {
                "split": split_name,
                "count": len(lengths),
                "mean": float(lengths.mean()),
                "min": int(lengths.min()),
                "p50": float(lengths.quantile(0.50)),
                "p95": float(lengths.quantile(0.95)),
                "p99": float(lengths.quantile(0.99)),
                "max": int(lengths.max()),
                "truncated_pct": float((lengths >= config.MAX_SEQUENCE_LENGTH).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows).set_index("split")
