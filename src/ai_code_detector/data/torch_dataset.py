"""PyTorch Dataset / DataLoader wrappers for the tokenized splits.

The rest of the project uses HuggingFace ``datasets`` as the canonical on-disk
format, but PyTorch training loops expect :class:`torch.utils.data.Dataset`
instances. This module bridges the two:

* :class:`CodeClassificationDataset` is a thin, type-checked wrapper that
  yields per-sample dicts (`input_ids`, `attention_mask`, `labels`).
* :func:`build_dataloader` constructs a :class:`~torch.utils.data.DataLoader`
  with the HuggingFace :class:`~transformers.DataCollatorWithPadding`, which
  dynamically pads each batch to its longest sequence (much faster than
  padding every sample to 512 up front).

The wrapper is intentionally HF-compatible: training code can use either
``CodeClassificationDataset(hf_dataset)`` or ``hf_dataset.with_format("torch")``
interchangeably.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset
    from transformers import PreTrainedTokenizerBase


class CodeClassificationDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch Dataset over a tokenized HuggingFace split.

    Each item is a dict with:

    * ``input_ids`` -- LongTensor of token ids.
    * ``attention_mask`` -- LongTensor of 0/1 flags.
    * ``labels`` -- LongTensor scalar with the class index.

    We deliberately emit ``labels`` (plural) rather than ``label``: this is
    the key HuggingFace Trainer and most PyTorch classifier heads expect.
    """

    __slots__ = ("_hf_dataset", "_label_column")

    def __init__(self, hf_dataset: HFDataset, *, label_column: str = "label") -> None:
        if label_column not in hf_dataset.column_names:
            raise KeyError(
                f"Label column {label_column!r} not found in dataset columns "
                f"{hf_dataset.column_names}."
            )
        self._hf_dataset = hf_dataset
        self._label_column = label_column

    def __len__(self) -> int:
        return len(self._hf_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self._hf_dataset[index]
        # Tensors are created as lists here; the DataCollator converts to
        # padded tensors at batch time. Converting per-sample keeps memory
        # flat regardless of max_length.
        item: dict[str, torch.Tensor | int | list[int]] = {
            "input_ids": row["input_ids"],
            "attention_mask": row["attention_mask"],
            "labels": int(row[self._label_column]),
        }
        return item  # type: ignore[return-value]


def build_dataloader(
    dataset: CodeClassificationDataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Construct a DataLoader with dynamic padding.

    Args:
        dataset: The wrapped tokenized split.
        tokenizer: Tokenizer used for the collator's pad token lookup.
        batch_size: Per-step batch size.
        shuffle: Whether to shuffle each epoch (True for train, False otherwise).
        num_workers: Number of subprocess workers.
        pin_memory: Pin DataLoader batches into pinned memory. Defaults to
            True iff CUDA is available.

    Returns:
        A DataLoader that yields dicts with padded ``input_ids``,
        ``attention_mask``, and ``labels`` tensors.
    """
    # Imported lazily so this module is importable without transformers
    # resolved (e.g., during docs builds or standalone dataset unit tests).
    from transformers import DataCollatorWithPadding

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )


def batch_to_device(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move every tensor in ``batch`` onto ``device``.

    A small helper so training loops don't repeat ``.to(device)`` boilerplate
    for every key.
    """
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
