# 📂 data/

> **⚠️ IMPORTANT: Downloaded datasets must remain local and must NOT be committed to GitHub.**

---

## Data Policy

This directory is intentionally kept empty in the repository.

All datasets used in this project are:
1. **Downloaded programmatically** at runtime using the HuggingFace `datasets` library.
2. **Stored locally** on your machine only — they are excluded from version control via `.gitignore`.
3. **Never committed** to GitHub, even partially. Dataset files can be hundreds of megabytes or several gigabytes in size, which would make the repository unusable for others.

---

## Why Not Commit Data?

| Reason              | Details                                                                 |
|---------------------|-------------------------------------------------------------------------|
| **File size**       | SST-2 alone is ~7 MB compressed; full NLP datasets can be GBs          |
| **Licensing**       | Some datasets have usage restrictions; redistribution may not be allowed |
| **Reproducibility** | Downloading from the canonical source ensures you always get the right version |
| **GitHub limits**   | GitHub blocks files over 100 MB and discourages large binary files      |

---

## Dataset Used: SST-2 (Stanford Sentiment Treebank)

| Property        | Value                                              |
|-----------------|----------------------------------------------------|
| **Source**      | [HuggingFace Hub — `glue` / `sst2`](https://huggingface.co/datasets/glue) |
| **Task**        | Binary sentiment classification (Positive / Negative) |
| **Train size**  | ~67,000 sentences                                  |
| **Val size**    | ~872 sentences                                     |
| **Test size**   | ~1,821 sentences (labels withheld — we use val)    |
| **Language**    | English                                            |

### How to Download

```python
from datasets import load_dataset

dataset = load_dataset("glue", "sst2")
# Data is automatically cached in ~/.cache/huggingface/datasets/
# You do NOT need to save anything to the data/ folder manually.
```

The HuggingFace library caches downloads automatically. If you want to redirect
the cache into this project folder (e.g., for offline use), you can set:

```bash
# Windows PowerShell
$env:HF_HOME = "D:\nlp-text-robustness\.cache"

# macOS / Linux
export HF_HOME="/path/to/nlp-text-robustness/.cache"
```

> The `.cache/` directory is already excluded in `.gitignore`.

---

## Directory Structure (when populated locally)

```
data/
├── README.md          ← This file (committed)
├── raw/               ← Raw downloaded files, if saved manually (git-ignored)
├── processed/         ← Preprocessed / corrupted versions (git-ignored)
└── cache/             ← Manual cache directory, if used (git-ignored)
```

---

## Adding New Datasets

If you add a new dataset in future experiments:
1. Load it via `datasets.load_dataset(...)` whenever possible.
2. If you must save files locally, store them in `data/raw/` (already git-ignored).
3. Document the dataset name, source URL, license, and size in this README.
4. **Never** `git add` a data file.
