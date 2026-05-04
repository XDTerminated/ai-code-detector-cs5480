"""Inference helpers: run a trained classifier and return aligned arrays.

The output of :func:`predict` is a :class:`PredictionResult` containing:

* ``probabilities``  -- shape ``(N,)`` of P(label=1).
* ``predictions``    -- shape ``(N,)`` of hard 0/1 predictions at the
  configured threshold.
* ``labels``         -- shape ``(N,)`` of ground-truth labels.

Returning all three in a single object guarantees they stay aligned -- the
single most common bug in evaluation code is shuffling labels and predictions
into different orderings.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm

from ai_code_detector import config
from ai_code_detector.data.torch_dataset import batch_to_device
from ai_code_detector.models.classifier import CodeBertBinaryClassifier

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PredictionResult:
    """Prediction outputs for one evaluation pass, all aligned by index."""

    probabilities: np.ndarray
    predictions: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        n = self.labels.shape[0]
        if not (self.probabilities.shape[0] == self.predictions.shape[0] == n):
            raise ValueError(
                f"PredictionResult arrays misaligned: "
                f"probs={self.probabilities.shape}, "
                f"preds={self.predictions.shape}, "
                f"labels={self.labels.shape}."
            )


@torch.no_grad()
def predict(
    model: CodeBertBinaryClassifier,
    dataloader: Iterable[dict],
    device: torch.device,
    *,
    threshold: float = config.DECISION_THRESHOLD,
    show_progress: bool = True,
    desc: str = "predicting",
) -> PredictionResult:
    """Run ``model`` over ``dataloader`` and return aligned probs/preds/labels."""
    model.eval()
    model.to(device)

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    iterator = dataloader
    if show_progress:
        iterator = tqdm(dataloader, desc=desc, leave=False)

    for batch in iterator:
        batch = batch_to_device(batch, device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        probs = torch.sigmoid(out.logits).detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)

    probabilities = np.concatenate(all_probs).astype(np.float64)
    labels_arr = np.concatenate(all_labels).astype(np.int64)
    predictions = (probabilities >= threshold).astype(np.int64)

    return PredictionResult(
        probabilities=probabilities,
        predictions=predictions,
        labels=labels_arr,
    )
