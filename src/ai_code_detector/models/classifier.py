"""Binary CodeBERT classifier for AI-vs-human Python code detection.

Architecture (matching the project proposal verbatim)
-----------------------------------------------------

    CodeBERT encoder (RobertaModel)
        -> [CLS]-token hidden state                  (B, 768)
        -> Dropout                                   (B, 768)
        -> Linear(hidden_size, 1)                    (B, 1)
        -> sigmoid (inference) / BCEWithLogitsLoss (training)

The model returns *raw logits* rather than probabilities so the training loop
can pair them with :class:`torch.nn.BCEWithLogitsLoss`, which fuses sigmoid
and binary cross-entropy in a numerically stable way (better gradients than
``log(sigmoid(x))``). At inference time, callers apply :func:`torch.sigmoid`
themselves.

The single-logit head (rather than a 2-class softmax head) follows the
proposal's wording: "fully connected layer with a sigmoid activation function
... binary cross-entropy loss".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, PretrainedConfig

from ai_code_detector import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClassifierForwardOutput:
    """Structured return value of :meth:`CodeBertBinaryClassifier.forward`.

    Attributes:
        logits: Raw, unbounded logits of shape ``(batch,)``. Apply
            :func:`torch.sigmoid` to recover probabilities.
        loss: BCE loss when ``labels`` were provided, otherwise ``None``.
    """

    logits: torch.Tensor
    loss: torch.Tensor | None


class CodeBertBinaryClassifier(nn.Module):
    """CodeBERT-based binary classifier with a single-logit sigmoid head.

    The encoder is loaded from the HuggingFace hub via :func:`transformers.AutoModel`,
    which gives us a vanilla :class:`~transformers.RobertaModel` for
    ``microsoft/codebert-base``. We deliberately do *not* use
    :class:`~transformers.AutoModelForSequenceClassification`: that class wires
    in a 2-logit softmax head and CrossEntropyLoss, whereas the proposal
    specifies a single-logit sigmoid head with BCE loss.
    """

    def __init__(
        self,
        model_name: str = config.MODEL_NAME,
        *,
        dropout_prob: float = config.CLASSIFIER_DROPOUT,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name

        cache = cache_dir if cache_dir is not None else str(config.HF_CACHE_DIR)
        self.encoder: nn.Module = AutoModel.from_pretrained(model_name, cache_dir=cache)

        encoder_config: PretrainedConfig = self.encoder.config  # type: ignore[assignment]
        hidden_size: int = encoder_config.hidden_size

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)

        # BCEWithLogitsLoss is the numerically-stable fusion of sigmoid + BCE
        # and is the canonical loss for single-logit binary classifiers.
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Init the head with small random weights so the encoder dominates
        # at the very first step (RoBERTa's recommended init for new heads).
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

        logger.info(
            "Initialized CodeBertBinaryClassifier(model=%s, hidden_size=%d, dropout=%.2f)",
            model_name,
            hidden_size,
            dropout_prob,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> ClassifierForwardOutput:
        """Run a forward pass.

        Args:
            input_ids: ``(batch, seq_len)`` token id tensor.
            attention_mask: ``(batch, seq_len)`` 0/1 padding mask.
            labels: Optional ``(batch,)`` integer labels in ``{0, 1}``. When
                provided, the BCE loss is computed and returned.

        Returns:
            A :class:`ClassifierForwardOutput` with ``logits`` of shape
            ``(batch,)`` and (if labels were given) ``loss``.
        """
        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # The first position of last_hidden_state is the [CLS] token (in
        # RoBERTa terminology, <s>). It is the canonical sentence-level
        # representation for classification fine-tuning.
        cls_hidden = encoder_out.last_hidden_state[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)
        logits = self.classifier(cls_hidden).squeeze(-1)  # (B,)

        loss: torch.Tensor | None = None
        if labels is not None:
            # BCEWithLogitsLoss expects float targets; cast once here.
            loss = self.loss_fn(logits, labels.to(logits.dtype))

        return ClassifierForwardOutput(logits=logits, loss=loss)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return P(label=1 | x) for each sample (no gradient)."""
        self.eval()
        out = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        return torch.sigmoid(out.logits)

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        """Total parameter count, optionally restricted to trainable params."""
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)
