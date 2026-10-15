# Robustness of Sentiment Classification Models Under Character-Level Text Corruption

> A beginner-friendly NLP research project exploring how well sentiment classifiers hold up when input text is deliberately corrupted at the character level.

---

## 🔬 Research Question

**How robust are pre-trained and classical sentiment classification models when input text is subjected to character-level corruption (e.g., typos, character swaps, deletions, insertions)?**

Modern NLP models achieve impressive accuracy on clean benchmark datasets, but real-world text is messy. Users make typos, autocorrect introduces errors, and adversarial actors deliberately obfuscate text. This project investigates whether popular sentiment classifiers degrade gracefully — or catastrophically — under controlled character-level noise.

---

## 📁 Project Structure

```
nlp-text-robustness/
├── README.md               ← You are here
├── .gitignore              ← Excludes data, checkpoints, and secrets
├── requirements.txt        ← Python dependencies
├── data/
│   └── README.md           ← Data policy and sources
├── notebooks/
│   └── README.md           ← Notebook index and usage guide
├── src/
│   └── __init__.py         ← Source package initializer
├── results/
│   └── README.md           ← Results storage policy
└── tests/
    └── README.md           ← Testing guide
```

---

## 🧪 Planned Experiments

### Corruption Types
We will implement the following character-level perturbation strategies:

| Corruption Type       | Description                                          | Example                        |
|-----------------------|------------------------------------------------------|--------------------------------|
| **Random deletion**   | Randomly drop characters from words                  | `"happy"` → `"hppy"`          |
| **Random insertion**  | Insert random characters at random positions         | `"great"` → `"grxeat"`        |
| **Character swap**    | Swap two adjacent characters in a word               | `"movie"` → `"mvie"`          |
| **Keyboard proximity**| Replace a character with a nearby key on a keyboard  | `"love"` → `"loge"`           |
| **Random substitution**| Replace a character with a random one               | `"bad"` → `"bzd"`             |
| **Repeated characters**| Repeat a random character multiple times            | `"good"` → `"goood"`          |

### Corruption Intensity Levels
Each corruption type will be tested at varying noise levels (percentage of characters affected):
- **0%** — clean baseline
- **5%**, **10%**, **20%**, **30%**, **50%** — progressive corruption

### Models Under Evaluation
1. **Logistic Regression (TF-IDF)** — Classical baseline
2. **Naive Bayes (TF-IDF)** — Classical baseline
3. **DistilBERT** *(added in Week 3)* — Transformer-based model
4. **RoBERTa** *(added in Week 4)* — Larger transformer for comparison

### Dataset
- **SST-2** (Stanford Sentiment Treebank) from the HuggingFace `datasets` library
- Binary classification: Positive / Negative

---

## 📊 Evaluation Metrics

| Metric          | Why It Matters                                                              |
|-----------------|-----------------------------------------------------------------------------|
| **Accuracy**    | Primary metric; overall fraction of correct predictions                     |
| **F1 Score**    | Balances precision and recall; robust to class imbalance                    |
| **Robustness Drop** | `Accuracy(clean) - Accuracy(corrupted)` — measures sensitivity to noise |
| **Degradation Curve** | Plot of accuracy vs. corruption intensity for each model             |
| **Consistency** | Whether the model agrees with itself on semantically identical noisy inputs |

---

## 🗓️ 30-Day Research Roadmap

### Week 1 — Foundation (Days 1–7)
- [x] **Day 1**: Initialize repository, set up project structure
- [ ] **Day 2**: Write and test character-level corruption utilities (`src/corruption.py`)
- [ ] **Day 3**: Load and explore SST-2 dataset (`notebooks/01_data_exploration.ipynb`)
- [ ] **Day 4**: Implement TF-IDF feature extraction pipeline
- [ ] **Day 5**: Train and evaluate Logistic Regression and Naive Bayes baselines on clean data
- [ ] **Day 6**: Run baselines on all corruption types at all intensity levels
- [ ] **Day 7**: Visualize baseline degradation curves; write summary notes

### Week 2 — Analysis & Hardening (Days 8–14)
- [ ] **Day 8**: Deep-dive error analysis — which corruption types hurt most?
- [ ] **Day 9**: Investigate per-class robustness (are positive/negative reviews affected differently?)
- [ ] **Day 10**: Attempt simple mitigation: spell-checking preprocessing
- [ ] **Day 11**: Re-evaluate baselines with spell-check preprocessing
- [ ] **Day 12**: Write `tests/` unit tests for corruption functions
- [ ] **Day 13**: Refactor `src/` into clean, importable modules
- [ ] **Day 14**: Mid-project checkpoint — document findings so far in `results/`

### Week 3 — Transformers (Days 15–21)
- [ ] **Day 15**: Add `torch` and `transformers` to requirements; set up GPU/CPU inference
- [ ] **Day 16**: Load pretrained DistilBERT and run inference on clean SST-2
- [ ] **Day 17**: Evaluate DistilBERT on all corruption types and intensity levels
- [ ] **Day 18**: Compare DistilBERT vs. classical baselines — generate comparison plots
- [ ] **Day 19**: Investigate attention weights on corrupted inputs (qualitative analysis)
- [ ] **Day 20**: Load RoBERTa; repeat evaluation
- [ ] **Day 21**: Transformer robustness summary; document findings

### Week 4 — Writeup & Polish (Days 22–30)
- [ ] **Day 22**: Compile all experiment results into `results/`
- [ ] **Day 23**: Create final comparison table and all publication-ready plots
- [ ] **Day 24**: Write `notebooks/05_final_analysis.ipynb` — end-to-end reproducible notebook
- [ ] **Day 25**: Draft research summary (Introduction, Methods, Results, Conclusion)
- [ ] **Day 26**: Peer review draft; revise figures
- [ ] **Day 27**: Write reproducibility guide (how to run all experiments from scratch)
- [ ] **Day 28**: Final test — clone repo fresh and run all notebooks in order
- [ ] **Day 29**: Polish README, add badges, finalize `.gitignore`
- [ ] **Day 30**: 🎉 Project complete — share on GitHub

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/nlp-text-robustness.git
cd nlp-text-robustness
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter
```bash
jupyter notebook
```

---

## 📌 Key Design Decisions

- **No data in the repo** — Datasets are downloaded programmatically via HuggingFace `datasets` and stored locally. See `data/README.md`.
- **Modular `src/`** — All reusable logic (corruption, evaluation, plotting) lives in `src/` so notebooks stay clean.
- **Reproducibility first** — Every experiment is seeded and documented so results can be independently verified.
- **Incremental complexity** — Classical models first, transformers later. Understand before scaling up.

---

## 📜 License

This project is for educational and research purposes. See `LICENSE` for details (to be added).

---

*Started: October 2026 | Status: 🟡 In Progress — Week 1*
