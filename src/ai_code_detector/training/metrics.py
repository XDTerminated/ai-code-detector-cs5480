"""Classification metrics for binary AI-vs-human code detection.

Provides:

* :func:`compute_classification_metrics` -- the canonical metric bundle the
  proposal asks for: accuracy, per-class precision/recall/F1, macro-F1,
  AUC-ROC, and the confusion matrix (raw counts).
* :class:`ClassificationMetrics` -- typed container that round-trips through
  JSON via :meth:`to_dict` / :meth:`from_dict` so the same shape can be saved
  to disk, logged, and rebuilt for reporting.

The implementation is a thin wrapper over scikit-learn so the math is
audited and the failure modes are well-understood.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ai_code_detector import config


@dataclass(slots=True)
class ClassificationMetrics:
    """Bundle of metrics for one evaluation run."""

    accuracy: float
    macro_f1: float
    auc_roc: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]] = field(default_factory=list)
    support: dict[str, int] = field(default_factory=dict)
    threshold: float = config.DECISION_THRESHOLD
    n_samples: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict) -> ClassificationMetrics:
        return cls(**payload)

    @classmethod
    def from_json(cls, path: Path) -> ClassificationMetrics:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def pretty(self) -> str:
        """Return a human-readable multi-line summary."""
        lines = [
            f"n_samples       = {self.n_samples}",
            f"threshold       = {self.threshold:.3f}",
            f"accuracy        = {self.accuracy:.4f}",
            f"macro_f1        = {self.macro_f1:.4f}",
            f"auc_roc         = {self.auc_roc:.4f}",
            "per-class:",
        ]
        for class_name, scores in self.per_class.items():
            lines.append(
                f"  {class_name:6s}  precision={scores['precision']:.4f}  "
                f"recall={scores['recall']:.4f}  f1={scores['f1']:.4f}  "
                f"support={int(scores['support'])}"
            )
        cm = np.array(self.confusion_matrix)
        if cm.size:
            lines.append("confusion_matrix (rows=true, cols=pred):")
            class_labels = list(self.per_class.keys())
            header = "         " + "  ".join(f"{c:>6s}" for c in class_labels)
            lines.append(header)
            for label, row in zip(class_labels, cm, strict=True):
                lines.append(f"  {label:6s} " + "  ".join(f"{v:>6d}" for v in row))
        return "\n".join(lines)


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_proba: Sequence[float] | np.ndarray,
    *,
    threshold: float = config.DECISION_THRESHOLD,
    class_names: Iterable[str] | None = None,
) -> ClassificationMetrics:
    """Compute the proposal's full metric bundle.

    Args:
        y_true: Ground-truth labels in ``{0, 1}``.
        y_proba: Predicted P(label=1). Must be on the same indexing as ``y_true``.
        threshold: Probability cutoff used to derive hard predictions.
        class_names: Names for class 0 and class 1 (in that order). Defaults
            to the values from :data:`ai_code_detector.config.LABEL_NAMES`.

    Returns:
        A populated :class:`ClassificationMetrics`.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_proba_arr = np.asarray(y_proba, dtype=np.float64)
    if y_true_arr.shape != y_proba_arr.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true_arr.shape}, y_proba={y_proba_arr.shape}.")
    if y_true_arr.size == 0:
        raise ValueError("Cannot compute metrics on an empty input.")

    y_pred_arr = (y_proba_arr >= threshold).astype(np.int64)

    if class_names is None:
        class_names = (config.LABEL_NAMES[0], config.LABEL_NAMES[1])
    class_names = list(class_names)
    if len(class_names) != 2:
        raise ValueError(f"class_names must have length 2; got {class_names!r}.")

    # Accuracy and macro-F1
    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )
    macro_f1 = float(np.mean(f1))

    # AUC-ROC requires both classes present in y_true; guard with a clear error
    # because the typical cause is an accidentally-tiny eval set.
    if len(np.unique(y_true_arr)) < 2:
        raise ValueError(
            "AUC-ROC is undefined when y_true contains only one class. "
            "Ensure the evaluation split is stratified and non-trivial."
        )
    auc_roc = float(roc_auc_score(y_true_arr, y_proba_arr))

    # Confusion matrix with the row/col order pinned to [0, 1] so plots and
    # JSON are deterministic regardless of which classes the model predicted.
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])

    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(2)
    }
    support_dict = {class_names[i]: int(support[i]) for i in range(2)}

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        auc_roc=auc_roc,
        per_class=per_class,
        confusion_matrix=cm.tolist(),
        support=support_dict,
        threshold=threshold,
        n_samples=int(y_true_arr.size),
    )
