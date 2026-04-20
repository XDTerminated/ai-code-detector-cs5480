# AI-Written Python Code Detection (CS5480)

Binary classification of Python source code as either **human-written** or
**AI-generated**, using a fine-tuned [CodeBERT](https://huggingface.co/microsoft/codebert-base)
model on the [HumanVSAI_CodeDataset](https://doi.org/10.17632/kjh95n54f8.1).

**Team:** Sayam Gupta, Kyle Marut, Karthik Thota, Aria Varwig.

---

## Project layout

```
ai-code-detector-cs5480/
├── data/
│   ├── raw/          # extracted HumanVSAI_CodeDataset (gitignored)
│   ├── interim/      # filtered Python-only subset (gitignored)
│   └── processed/    # splits + tokenized Arrow datasets (gitignored)
├── reports/
│   └── figures/      # EDA plots
├── scripts/
│   ├── prepare_dataset.py   # Phase 1: load → normalize → filter → split
│   ├── analyze_dataset.py   # Phase 1: EDA tables + plots
│   └── tokenize_dataset.py  # Phase 2: CodeBERT tokenization
├── src/
│   └── ai_code_detector/
│       ├── config.py               # paths, constants, seeds, model name
│       ├── logging_utils.py
│       ├── data/
│       │   ├── loading.py          # raw-file IO + schema normalization
│       │   ├── filtering.py        # Python-only filter, dedupe, clean
│       │   ├── splitting.py        # stratified train/val/test split
│       │   └── torch_dataset.py    # PyTorch Dataset + DataLoader builders
│       └── features/
│           └── tokenization.py     # CodeBERT tokenization
├── pyproject.toml
└── uv.lock
```

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management
and Python 3.13.

```bash
# Install dependencies (creates .venv, installs package in editable mode).
uv sync
```

All subsequent commands should be invoked with `uv run` so they execute
inside the project virtualenv.

---

## Downloading the dataset

The HumanVSAI_CodeDataset is hosted on Mendeley Data and requires a browser
to download (it's behind a JS-rendered download button; there is no stable
programmatic URL).

1. Open <https://data.mendeley.com/datasets/kjh95n54f8/1>.
2. Click **Download All** (the resulting file is a `.zip`).
3. Extract the archive into `data/raw/` so the tabular files sit directly
   under that directory:

   ```
   data/raw/
   ├── <dataset-file-1>.csv
   ├── <dataset-file-2>.json
   └── ...
   ```

The `data/raw/` directory is gitignored. The loader auto-detects and
concatenates every file with a `.csv`, `.tsv`, `.json`, `.jsonl`, `.ndjson`,
or `.parquet` suffix, so the exact file names and nesting do not matter.

If the dataset's column names differ from what the loader expects, extend
the `*_CANDIDATES` tuples in [`src/ai_code_detector/config.py`](src/ai_code_detector/config.py)
— you do not need to touch the loading code.

---

## Running the pipeline

### Phase 1 — prepare the dataset

```bash
uv run python scripts/prepare_dataset.py
```

This will:

1. Load every supported file in `data/raw/`.
2. Normalize onto the canonical `(code, label, language)` schema.
3. Filter to Python only, drop empty snippets and duplicates.
4. Produce a stratified 70/15/15 train/validation/test split with a fixed
   random seed (42).
5. Write Parquet artifacts to `data/interim/python_only.parquet` and
   `data/processed/splits/{train,validation,test}.parquet`.

Class balance is printed for each stage so you can spot class drift.

### Phase 1 (cont.) — exploratory data analysis

```bash
uv run python scripts/analyze_dataset.py
```

Produces tables in the console (class balance per split, length
distributions per class, duplicate counts) and PNG plots under
`reports/figures/`:

- `class_balance.png`
- `distribution_n_chars.png`
- `distribution_n_lines.png`
- `distribution_n_whitespace_tokens.png`

### Phase 2 — tokenize with CodeBERT

```bash
uv run python scripts/tokenize_dataset.py
```

This will:

1. Load the Phase 1 split Parquet files.
2. Load the `microsoft/codebert-base` fast tokenizer (cached under `.cache/`).
3. Tokenize every split with truncation at 512 tokens (no padding — padding is
   applied dynamically at batch time during training via
   `DataCollatorWithPadding`).
4. Save a HuggingFace `DatasetDict` to
   `data/processed/tokenized/`, loadable with `datasets.load_from_disk(...)`.

A sequence-length summary is printed so you can confirm the truncation rate
is acceptable before starting training.

---

## Reproducibility

- Every stochastic step (split, shuffling) uses `config.RANDOM_SEED = 42`.
- `uv.lock` pins every transitive dependency.
- Intermediate artifacts are written to disk so each phase can be re-run
  independently of the others.

---

## Next steps (Phases 3+)

- Phase 3: CodeBERT classifier + training loop (`src/ai_code_detector/models/`,
  `scripts/train.py`).
- Phase 4: Evaluation (accuracy, per-class P/R/F1, AUC-ROC, confusion matrix).
- Phase 5: Hyperparameter tuning.
- Phase 6: Ablations (input format, snippet length) and error analysis.
- Phase 7: Final report and poster.
