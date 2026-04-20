"""Raw-dataset loading and schema normalization.

The HumanVSAI_CodeDataset is distributed as generic tabular files (CSV, JSON,
JSONL, or Parquet) on Mendeley. We do not hardcode a specific schema because
the upstream dataset card does not publish one; instead we probe for any of a
small set of candidate column names (see :mod:`ai_code_detector.config`) and
normalize into the canonical triple ``(code, label, language)``.

Two input shapes are supported:

1. **Long-form** (one row per sample): a label column and a language column
   exist. This is the most common case.
2. **Paired-form** (one row = one human/AI pair): columns ``human_code`` and
   ``ai_code`` appear side by side. We explode such rows into two long-form
   rows so downstream code never has to special-case the layout.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ai_code_detector import config

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES: frozenset[str] = frozenset(
    {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".parquet"}
)


# ---------------------------------------------------------------------------
# File discovery and single-file loading
# ---------------------------------------------------------------------------
def discover_raw_files(raw_dir: Path) -> list[Path]:
    """Return all supported tabular files under ``raw_dir`` (recursive).

    Args:
        raw_dir: Directory containing the extracted HumanVSAI_CodeDataset.

    Returns:
        Sorted list of data-file paths. Empty if none are found.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory does not exist: {raw_dir}. "
            "See README.md for download instructions."
        )

    files = sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
    )
    return files


def load_tabular_file(path: Path) -> pd.DataFrame:
    """Load a single tabular file into a DataFrame, dispatching on suffix.

    Args:
        path: Path to a CSV, TSV, JSON, JSONL/NDJSON, or Parquet file.

    Returns:
        The file's contents as a DataFrame.

    Raises:
        ValueError: If the file suffix is not supported.
    """
    suffix = path.suffix.lower()
    logger.debug("Loading %s (suffix=%s)", path, suffix)

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".json":
        # A .json file may hold either a single JSON document or one-per-line
        # records. Try standard parsing first and fall back to ``lines=True``.
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file suffix {suffix!r} for {path}")


def load_raw_dataset(raw_dir: Path = config.RAW_DATA_DIR) -> pd.DataFrame:
    """Load and concatenate every supported tabular file in ``raw_dir``.

    Each source path is attached as a ``__source_file`` column so that
    provenance is preserved through later stages (useful for debugging and
    error analysis).

    Args:
        raw_dir: Directory containing the extracted raw dataset files.

    Returns:
        Concatenated DataFrame with all raw rows.

    Raises:
        FileNotFoundError: If ``raw_dir`` is empty or contains no supported
            files.
    """
    files = discover_raw_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No supported data files found under {raw_dir}. "
            f"Expected any of: {sorted(_SUPPORTED_SUFFIXES)}. "
            "Download and extract the HumanVSAI_CodeDataset per README.md."
        )

    logger.info("Discovered %d raw file(s) under %s", len(files), raw_dir)
    frames: list[pd.DataFrame] = []
    for path in files:
        df = load_tabular_file(path)
        df["__source_file"] = path.relative_to(raw_dir).as_posix()
        logger.info("  %s -> %d rows, %d columns", path.name, len(df), df.shape[1])
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined raw dataset: %d rows, columns=%s", len(combined), list(combined.columns))
    return combined


# ---------------------------------------------------------------------------
# Schema normalization
# ---------------------------------------------------------------------------
def _resolve_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    role: str,
    *,
    required: bool = True,
) -> str | None:
    """Return the first column in ``candidates`` that exists in ``df``.

    The comparison is case-insensitive so columns named ``Code`` or ``CODE``
    still match.

    Args:
        df: Source DataFrame.
        candidates: Candidate column names, in priority order.
        role: Human-readable name of the column's semantic role, only used to
            build error messages.
        required: If True, raise when no candidate matches. Otherwise return
            ``None``.

    Returns:
        The matching column name (in its original casing) or ``None``.

    Raises:
        KeyError: If ``required`` is True and no candidate matches.
    """
    lower_to_original = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    if required:
        raise KeyError(
            f"Could not locate a {role} column. Looked for any of "
            f"{tuple(candidates)} in columns {list(df.columns)}. "
            "Add your column name to the relevant *_CANDIDATES tuple in "
            "ai_code_detector.config."
        )
    return None


def _encode_label(value: object) -> int:
    """Map a raw label value to the canonical 0/1 encoding.

    Accepts ints, bools, and the string aliases declared in
    :mod:`ai_code_detector.config`. Case and surrounding whitespace are ignored
    for strings.

    Raises:
        ValueError: If the value cannot be mapped.
    """
    if isinstance(value, bool):
        return config.LABEL_AI if value else config.LABEL_HUMAN
    if isinstance(value, (int, float)):
        as_int = int(value)
        if as_int in (config.LABEL_HUMAN, config.LABEL_AI):
            return as_int
        raise ValueError(f"Numeric label {value!r} is outside of {{0, 1}}.")

    normalized = str(value).strip().lower()
    if normalized in config.HUMAN_LABEL_ALIASES:
        return config.LABEL_HUMAN
    if normalized in config.AI_LABEL_ALIASES:
        return config.LABEL_AI
    raise ValueError(
        f"Unrecognized label value {value!r}. Extend HUMAN_LABEL_ALIASES or "
        "AI_LABEL_ALIASES in ai_code_detector.config."
    )


def _normalize_language(value: object) -> str:
    """Lowercase and canonicalize a language identifier."""
    return str(value).strip().lower()


def _explode_paired_format(df: pd.DataFrame) -> pd.DataFrame | None:
    """Convert a paired-format DataFrame to long-form, or return None.

    A paired row has both a human_code and ai_code column; we emit two rows
    per source row, one per class. If the columns are not present, this
    returns ``None`` so callers can fall through to the long-form path.
    """
    lowered = {col.lower(): col for col in df.columns}
    human_col = next(
        (lowered[c] for c in ("human_code", "human", "human_written_code") if c in lowered),
        None,
    )
    ai_col = next(
        (lowered[c] for c in ("ai_code", "ai", "ai_generated_code", "llm_code") if c in lowered),
        None,
    )
    if human_col is None or ai_col is None:
        return None

    logger.info(
        "Detected paired-format columns (%s, %s); exploding to long form.",
        human_col,
        ai_col,
    )

    # Language column, if present, applies to both members of the pair.
    lang_col = _resolve_column(df, config.LANGUAGE_COLUMN_CANDIDATES, "language", required=False)

    human_rows = pd.DataFrame(
        {
            config.CODE_COLUMN: df[human_col].astype(str),
            config.LABEL_COLUMN: config.LABEL_HUMAN,
        }
    )
    ai_rows = pd.DataFrame(
        {
            config.CODE_COLUMN: df[ai_col].astype(str),
            config.LABEL_COLUMN: config.LABEL_AI,
        }
    )
    if lang_col is not None:
        human_rows[config.LANGUAGE_COLUMN] = df[lang_col].map(_normalize_language)
        ai_rows[config.LANGUAGE_COLUMN] = df[lang_col].map(_normalize_language)
    else:
        human_rows[config.LANGUAGE_COLUMN] = pd.NA
        ai_rows[config.LANGUAGE_COLUMN] = pd.NA

    return pd.concat([human_rows, ai_rows], ignore_index=True)


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Project the raw DataFrame onto the canonical (code, label, language) schema.

    The returned DataFrame has exactly three columns in this order:
    ``code`` (str), ``label`` (int ∈ {0, 1}), ``language`` (str).

    Args:
        df: Raw DataFrame returned by :func:`load_raw_dataset`.

    Returns:
        Canonicalized DataFrame.
    """
    # Paired format gets special handling up front.
    exploded = _explode_paired_format(df)
    if exploded is not None:
        return _coerce_canonical_dtypes(exploded)

    code_col = _resolve_column(df, config.CODE_COLUMN_CANDIDATES, "code")
    label_col = _resolve_column(df, config.LABEL_COLUMN_CANDIDATES, "label")
    language_col = _resolve_column(
        df, config.LANGUAGE_COLUMN_CANDIDATES, "language", required=False
    )

    out = pd.DataFrame(
        {
            config.CODE_COLUMN: df[code_col].astype(str),
            config.LABEL_COLUMN: df[label_col].map(_encode_label),
        }
    )
    if language_col is not None:
        out[config.LANGUAGE_COLUMN] = df[language_col].map(_normalize_language)
    else:
        # If no language column is present we cannot filter; mark as unknown
        # and let the caller decide what to do.
        logger.warning(
            "No language column found; all rows will be tagged as 'unknown'. "
            "This will prevent Python-only filtering downstream."
        )
        out[config.LANGUAGE_COLUMN] = "unknown"

    return _coerce_canonical_dtypes(out)


def _coerce_canonical_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce dtypes on the canonical schema (in-place then returned)."""
    df[config.CODE_COLUMN] = df[config.CODE_COLUMN].astype(str)
    df[config.LABEL_COLUMN] = df[config.LABEL_COLUMN].astype("int64")
    df[config.LANGUAGE_COLUMN] = df[config.LANGUAGE_COLUMN].astype(str)
    return df[[config.CODE_COLUMN, config.LABEL_COLUMN, config.LANGUAGE_COLUMN]].copy()
