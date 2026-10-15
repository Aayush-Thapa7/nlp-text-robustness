# 📓 notebooks/

This directory contains all Jupyter notebooks for the `nlp-text-robustness` project.
Notebooks are the primary tool for exploration, visualization, and presenting results.

---

## Naming Convention

Notebooks are numbered to reflect the intended execution order:

```
notebooks/
├── 01_data_exploration.ipynb       ← Load SST-2, inspect distributions
├── 02_corruption_demo.ipynb        ← Demonstrate all corruption types visually
├── 03_baseline_experiments.ipynb   ← Train and evaluate classical models
├── 04_transformer_experiments.ipynb← DistilBERT & RoBERTa evaluation (Week 3+)
└── 05_final_analysis.ipynb         ← End-to-end comparison and final plots
```

> Notebooks are created progressively. Only notebook 01 onwards will exist after Week 1.

---

## Usage

1. Activate your virtual environment first:
   ```bash
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

2. Launch Jupyter from the **project root** (not from inside `notebooks/`):
   ```bash
   jupyter notebook
   ```
   Running from the root ensures that `import src` works correctly inside notebooks.

3. Run notebooks **in numerical order** for the first time — later notebooks may
   depend on outputs or knowledge from earlier ones.

---

## Ground Rules for Notebooks

| Rule | Why |
|------|-----|
| **Keep notebooks clean** | Clear all outputs before committing (`Kernel → Restart & Clear Output`) |
| **No heavy logic in notebooks** | Move reusable functions to `src/`; call them from notebooks |
| **One notebook per experiment** | Don't mix data exploration with model training in one notebook |
| **Document as you go** | Use Markdown cells to explain what you're doing and what you observe |
| **Seed all randomness** | `np.random.seed(42)` at the top of every notebook for reproducibility |

---

## Checkpoints

Jupyter automatically creates `.ipynb_checkpoints/` directories. These are
excluded from version control via `.gitignore` — do not commit them.
