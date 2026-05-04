"""Central project configuration.

All tunable paths, constants, and hyperparameters live here so that scripts and
library modules share a single source of truth. Importing code should reference
these constants by name rather than hardcoding values, which makes experiments
reproducible and sweeps trivial to implement.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
# PROJECT_ROOT is resolved relative to this file so scripts can be launched
# from any working directory without breaking path resolution.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# Concrete output artifacts for the Phase 1 / Phase 2 pipeline.
PYTHON_ONLY_PARQUET: Path = INTERIM_DATA_DIR / "python_only.parquet"
SPLIT_DIR: Path = PROCESSED_DATA_DIR / "splits"
TOKENIZED_DIR: Path = PROCESSED_DATA_DIR / "tokenized"

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------
# The HumanVSAI_CodeDataset does not publish a schema spec up front, so we
# accept any of these column names and normalize to a canonical schema.
# Extend these tuples if the raw dataset uses different names.
CODE_COLUMN_CANDIDATES: tuple[str, ...] = (
    "code",
    "source_code",
    "sample_code",
    "text",
    "snippet",
    "content",
    "sample",
    "source",
)
LABEL_COLUMN_CANDIDATES: tuple[str, ...] = (
    "label",
    "class",
    "is_ai",
    "ai_generated",
    "generated",
    "target",
    "y",
)
LANGUAGE_COLUMN_CANDIDATES: tuple[str, ...] = (
    "language",
    "lang",
    "programming_language",
    "prog_lang",
)

# Canonical column names we use throughout the project after normalization.
CODE_COLUMN: str = "code"
LABEL_COLUMN: str = "label"
LANGUAGE_COLUMN: str = "language"

# Canonical label encoding.
LABEL_HUMAN: int = 0
LABEL_AI: int = 1
LABEL_NAMES: dict[int, str] = {LABEL_HUMAN: "human", LABEL_AI: "ai"}

# String aliases we accept for each class when the raw dataset uses text labels.
HUMAN_LABEL_ALIASES: frozenset[str] = frozenset(
    {"human", "human-written", "human_written", "0", "false"}
)
AI_LABEL_ALIASES: frozenset[str] = frozenset(
    {"ai", "ai-generated", "ai_generated", "llm", "gpt", "chatgpt", "1", "true"}
)

# The proposal targets Python only. We accept common aliases defensively.
TARGET_LANGUAGE: str = "python"
PYTHON_LANGUAGE_ALIASES: frozenset[str] = frozenset({"python", "py", "python3"})

# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
# 70 / 15 / 15 stratified split. Fixed seed makes the split reproducible across
# machines and across experiment re-runs.
TRAIN_RATIO: float = 0.70
VALIDATION_RATIO: float = 0.15
TEST_RATIO: float = 0.15
RANDOM_SEED: int = 42

SPLIT_NAMES: tuple[str, str, str] = ("train", "validation", "test")

# ---------------------------------------------------------------------------
# Tokenization / model
# ---------------------------------------------------------------------------
# CodeBERT is a RoBERTa-base model pre-trained on CodeSearchNet, so the 512
# sequence length is the native upper bound. Keeping it as the default aligns
# with the CodeBERT paper and the project proposal.
MODEL_NAME: str = "microsoft/codebert-base"
MAX_SEQUENCE_LENGTH: int = 512

# HuggingFace datasets caches intermediate artifacts here so re-runs are fast
# and stay inside the repo (nothing leaks into ~/.cache).
HF_CACHE_DIR: Path = PROJECT_ROOT / ".cache" / "huggingface"

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
# Single-logit head (per the proposal): produces one scalar per sample which
# is fed into BCEWithLogitsLoss at training time and through a sigmoid at
# inference. This matches the proposal's "fully connected layer with sigmoid
# activation function ... binary cross-entropy loss" exactly.
NUM_CLASSES: int = 2
CLASSIFIER_DROPOUT: float = 0.1

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
# Defaults are taken from the original CodeBERT paper's fine-tuning recipe
# and the standard RoBERTa-classification recipe. Override on the CLI for
# sweeps.
DEFAULT_LEARNING_RATE: float = 2e-5
DEFAULT_WEIGHT_DECAY: float = 0.01
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_EVAL_BATCH_SIZE: int = 32
DEFAULT_NUM_EPOCHS: int = 3
DEFAULT_WARMUP_RATIO: float = 0.1
DEFAULT_GRADIENT_CLIP_NORM: float = 1.0
DEFAULT_EARLY_STOP_PATIENCE: int = 2  # epochs without val-loss improvement

# Decision threshold applied to sigmoid(logit) at inference. 0.5 is the
# Bayes-optimal threshold for a balanced loss; can be re-tuned on the
# validation set during evaluation.
DECISION_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Output directories for trained artifacts and reports
# ---------------------------------------------------------------------------
MODELS_DIR: Path = PROJECT_ROOT / "models"
BASELINE_MODEL_DIR: Path = MODELS_DIR / "baseline"
METRICS_DIR: Path = REPORTS_DIR / "metrics"
