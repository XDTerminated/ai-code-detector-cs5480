"""Plotting helpers for training curves and evaluation diagnostics.

All plotting functions take a destination :class:`~pathlib.Path` and write a
PNG. The matplotlib state is cleaned up before returning so calling many
plotters in a loop does not leak memory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve

from ai_code_detector import config

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="talk")


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def plot_training_curves(history: list[dict], out_path: Path) -> None:
    """Plot train + validation loss and validation accuracy/F1/AUC vs epoch."""
    if not history:
        logger.warning("Empty training history; skipping training-curves plot.")
        return

    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    val_acc = [r["val_metrics"]["accuracy"] for r in history]
    val_f1 = [r["val_metrics"]["macro_f1"] for r in history]
    val_auc = [r["val_metrics"]["auc_roc"] for r in history]

    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(14, 5))

    ax_loss.plot(epochs, train_loss, marker="o", label="train")
    ax_loss.plot(epochs, val_loss, marker="o", label="val")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("BCE loss")
    ax_loss.set_title("Loss vs epoch")
    ax_loss.legend()

    ax_score.plot(epochs, val_acc, marker="o", label="val accuracy")
    ax_score.plot(epochs, val_f1, marker="o", label="val macro-F1")
    ax_score.plot(epochs, val_auc, marker="o", label="val AUC-ROC")
    ax_score.set_xlabel("Epoch")
    ax_score.set_ylabel("Score")
    ax_score.set_ylim(0.0, 1.0)
    ax_score.set_title("Validation metrics vs epoch")
    ax_score.legend()

    fig.suptitle("Training curves")
    fig.tight_layout()
    _save(fig, out_path)


def plot_confusion_matrix(
    confusion: list[list[int]] | np.ndarray,
    out_path: Path,
    *,
    class_names: tuple[str, str] = (
        config.LABEL_NAMES[0],
        config.LABEL_NAMES[1],
    ),
    normalize: bool = False,
    title: str | None = None,
) -> None:
    """Render a confusion matrix as an annotated heatmap."""
    cm = np.asarray(confusion, dtype=np.float64 if normalize else np.int64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # Guard against rows with zero support (would otherwise produce NaN).
        cm = np.divide(cm, row_sums, where=row_sums > 0)

    fmt = ".2f" if normalize else "d"
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or ("Confusion matrix (normalized)" if normalize else "Confusion matrix"))
    fig.tight_layout()
    _save(fig, out_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_path: Path,
    *,
    auc_value: float | None = None,
    title: str = "ROC curve",
) -> None:
    """Plot the ROC curve with the AUC value in the legend."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}" if auc_value is not None else "ROC")
    # Random-classifier diagonal as a reference.
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.7, label="random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, out_path)


def plot_per_class_metrics(per_class: dict[str, dict[str, float]], out_path: Path) -> None:
    """Bar chart of per-class precision / recall / F1."""
    classes = list(per_class.keys())
    metrics = ["precision", "recall", "f1"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics))
    width = 0.35
    for offset, class_name in zip([-width / 2, width / 2], classes, strict=True):
        scores = [per_class[class_name][m] for m in metrics]
        bars = ax.bar(x + offset, scores, width=width, label=class_name)
        for bar, score in zip(bars, scores, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class precision / recall / F1")
    ax.legend(title="class")
    fig.tight_layout()
    _save(fig, out_path)
