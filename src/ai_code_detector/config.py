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
