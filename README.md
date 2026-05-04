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
├── models/                      # trained checkpoints (gitignored)
│   └── baseline/                # default training output
├── reports/
│   ├── figures/                 # EDA + evaluation plots
│   ├── metrics/                 # JSON metric bundles, error CSVs
│   └── results_baseline.md      # human-readable run summary
├── scripts/
│   ├── prepare_dataset.py       # Phase 1: load → normalize → filter → split
│   ├── analyze_dataset.py       # Phase 1: EDA tables + plots
│   ├── tokenize_dataset.py      # Phase 2: CodeBERT tokenization
│   ├── train.py                 # Phase 3: fine-tune CodeBERT
│   ├── evaluate.py              # Phase 4: test-set metrics + plots
│   ├── error_analysis.py        # Phase 5: misclassified-sample dump
│   ├── hyperparameter_search.py # Phase 4: small grid search
│   └── ablation_max_length.py   # Phase 5: max_length ablation
├── src/
│   └── ai_code_detector/
│       ├── config.py               # paths, constants, seeds, model + training defaults
│       ├── logging_utils.py
│       ├── data/
│       │   ├── loading.py          # raw-file IO + schema normalization
│       │   ├── filtering.py        # Python-only filter, dedupe, clean
│       │   ├── splitting.py        # stratified train/val/test split
│       │   └── torch_dataset.py    # PyTorch Dataset + DataLoader builders
│       ├── features/
│       │   └── tokenization.py     # CodeBERT tokenization
│       ├── models/
│       │   └── classifier.py       # CodeBERT + linear head + sigmoid (BCE)
│       ├── training/
│       │   ├── loop.py             # AdamW + linear warmup, val each epoch, early stop
│       │   ├── metrics.py          # accuracy, P/R/F1, AUC-ROC, confusion matrix
│       │   └── checkpoint.py       # safetensors save/load + tokenizer + history
│       └── evaluation/
│           ├── predict.py          # aligned (probs, preds, labels) inference
│           └── plots.py            # training curves, CM heatmap, ROC, per-class bars
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

### Phase 3 — fine-tune CodeBERT

```bash
uv run python scripts/train.py
```

Defaults (set in [`src/ai_code_detector/config.py`](src/ai_code_detector/config.py)):

| Hyperparameter        | Value           |
|-----------------------|-----------------|
| Optimizer             | AdamW           |
| Learning rate         | 2e-5            |
| Weight decay          | 0.01 (excluding bias / LayerNorm) |
| Warmup ratio          | 0.10 (linear warmup → linear decay) |
| Batch size (train)    | 16              |
| Batch size (eval)     | 32              |
| Epochs                | 3               |
| Gradient clip norm    | 1.0             |
| Classifier dropout    | 0.10            |
| Decision threshold    | 0.50            |
| Early-stop patience   | 2 epochs without val-loss improvement |
| Loss                  | `BCEWithLogitsLoss` (single-logit head per the proposal) |

The script writes the **best-by-validation-loss** checkpoint to
`models/baseline/`, including:

- `model.safetensors` — model weights.
- `training_config.json` — every hyperparameter used.
- `training_history.json` — per-epoch train/val loss + metrics.
- `tokenizer/` — the CodeBERT tokenizer (so inference is offline).

CPU note: a full 3-epoch run at the proposal's `max_length=512` takes
~2–3 hours on a modern 4-thread laptop. For development we re-tokenize at
`max_length=256` (drops ~10–15% of the longest snippets but cuts training
time to ~30–60 min):

```bash
uv run python scripts/tokenize_dataset.py --max-length 256 --out-dir data/processed/tokenized_l256
uv run python scripts/train.py --tokenized-dir data/processed/tokenized_l256
```

### Phase 4 — evaluate on the held-out test split

```bash
uv run python scripts/evaluate.py --checkpoint-dir models/baseline
```

Writes:

- `reports/metrics/baseline_test.json` — full metric bundle.
- `reports/figures/baseline_test_confusion_matrix{,_norm}.png` — raw and row-normalized.
- `reports/figures/baseline_test_roc_curve.png` — ROC curve with AUC.
- `reports/figures/baseline_test_per_class_metrics.png` — P/R/F1 bars.
- `reports/figures/baseline_training_curves.png` — loss + val metrics vs epoch.

### Phase 4 — small hyperparameter sweep (optional, expensive)

```bash
uv run python scripts/hyperparameter_search.py --configs 3
```

Each config is a full fine-tuning run, so this is opt-in. Results land in
`reports/metrics/sweep_results.csv` (sorted by val macro-F1) and per-config
checkpoints under `models/sweep/`.

### Phase 5 — error analysis

```bash
uv run python scripts/error_analysis.py --checkpoint-dir models/baseline
```

Writes:

- `reports/metrics/baseline_errors.csv` — top-N misclassified samples sorted
  by model confidence (most-confident-but-wrong first; these are the most
  diagnostic).
- `reports/metrics/baseline_error_summary.json` — aggregate stats: error
  rate per class, mean confidence on errors, length distribution comparison
  vs. correct predictions.

### Phase 5 — max-length ablation (optional, expensive)

```bash
uv run python scripts/ablation_max_length.py --lengths 128 256 512
```

One full fine-tune per length value. Output table at
`reports/metrics/ablation_max_length.csv`.

---

## Reproducibility

- Every stochastic step (split, shuffling) uses `config.RANDOM_SEED = 42`.
- `uv.lock` pins every transitive dependency.
- Intermediate artifacts are written to disk so each phase can be re-run
  independently of the others.

---

## Roadmap

- [x] **Phase 1** — load + clean + Python-only filter + stratified split.
- [x] **Phase 1 EDA** — class balance, code-length distributions, duplicates.
- [x] **Phase 2** — CodeBERT tokenization (HuggingFace `DatasetDict`).
- [x] **Phase 3** — CodeBERT fine-tuning loop with BCE loss + AdamW.
- [x] **Phase 4** — accuracy / per-class P/R/F1 / AUC-ROC / confusion matrix.
- [x] **Phase 4 (HP)** — opt-in grid search over LR / batch / epochs.
- [x] **Phase 5** — misclassification dump + length-stratified error stats.
- [x] **Phase 5 (ablation)** — `max_length` ∈ {128, 256, 512} ablation.
- [ ] Phase 6 — additional ablations (input format / comment-stripping).
- [ ] Phase 7 — final report and poster.
