# Robustness of Sentiment Classification Models Under Character-Level Text Corruption

A reproducible NLP research project investigating how character-level text corruption affects sentiment classification performance across traditional machine learning models and a fine-tuned Transformer.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Task](https://img.shields.io/badge/Task-Sentiment%20Analysis-purple.svg)](https://en.wikipedia.org/wiki/Sentiment_analysis)

## Overview

Sentiment classification systems often receive text containing typos, missing characters, inserted characters, or misspelled words. Such changes can affect the predictions made by machine learning models.

This project investigates the robustness of three sentiment classifiers under controlled character-level text corruption:

1. TF-IDF + Logistic Regression
2. TF-IDF + Linear Support Vector Machine (SVM)
3. Fine-tuned DistilBERT

The project also includes a Flask web application for testing sentiment predictions, experimenting with corrupted text, and viewing evaluation results.

**Research question:** How does character-level text corruption affect the accuracy and macro-F1 of sentiment classification models, and how do traditional machine learning models compare with a fine-tuned Transformer?

## Research Objectives

- Establish clean-text performance baselines for three classifiers.
- Evaluate deletion, insertion, substitution, and transposition corruption.
- Measure performance at corruption rates of 0%, 5%, 10%, and 20%.
- Compare accuracy, macro-F1, and accuracy degradation.
- Provide a web application for interactive exploration of model predictions.
- Make the experiments and results easier to reproduce.

## Dataset

The project uses the IMDb movie review dataset for binary sentiment classification.

| Property | Description |
|---|---|
| Dataset | IMDb movie reviews |
| Hugging Face dataset identifier | `stanfordnlp/imdb` |
| Training examples | 25,000 |
| Test examples | 25,000 |
| Task | Binary sentiment classification |
| Classes | Negative and positive |

The 50,000 labeled training and test reviews are used for model training and evaluation. The dataset also contains an unlabeled split, which is not the focus of these experiments.

Dataset source: [IMDb dataset on Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb)

Dataset files are not included in this repository; they are downloaded by the dataset-loading code.

## Models and Methodology

### 1. TF-IDF + Logistic Regression

The baseline pipeline converts reviews into TF-IDF features and uses Logistic Regression for binary sentiment classification.

### 2. TF-IDF + Linear SVM

A Linear Support Vector Machine is trained using TF-IDF features and evaluated on clean and corrupted reviews.

### 3. DistilBERT

`distilbert/distilbert-base-uncased` is fine-tuned for binary sentiment classification. The current training configuration uses a maximum sequence length of 256 tokens, batch size 8, learning rate `2e-5`, two epochs, and seed 42.

The reported DistilBERT experiment used GPU acceleration and mixed-precision training.

### Character-Level Corruption Strategies

| Strategy | Description |
|---|---|
| Deletion | Removes selected characters |
| Insertion | Adds characters at selected positions |
| Substitution | Replaces selected characters |
| Transposition | Swaps adjacent characters |

The experiments use corruption rates of **0%, 5%, 10%, and 20%**. A rate of 0% represents clean text.

The corruption utilities support seeded random behavior. For a valid model comparison, the same corrupted inputs should be evaluated across models, with the random seed and corruption implementation held consistent.

## Results

The following results were obtained in the current experiments. Accuracy is reported as a percentage. Accuracy drop is measured in percentage points relative to each model's clean accuracy.

### Clean-Test Performance

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Logistic Regression | 89.92% | 89.92% |
| Linear SVM | 90.25% | 90.25% |
| DistilBERT | 91.43% | 91.43% |

DistilBERT achieved the highest clean-test accuracy in these runs.

### Accuracy at 20% Character Corruption

| Corruption strategy | Logistic Regression | Linear SVM | DistilBERT |
|---|---:|---:|---:|
| Deletion | 75.93% | 71.56% | 55.40% |
| Insertion | 78.00% | 75.50% | 55.51% |
| Substitution | 76.12% | 73.14% | 55.51% |
| Transposition | 73.88% | 69.79% | 54.62% |

### Accuracy Drop at 20% Corruption

| Model | Deletion | Insertion | Substitution | Transposition |
|---|---:|---:|---:|---:|
| Logistic Regression | 13.99 pp | 11.92 pp | 13.80 pp | 16.04 pp |
| Linear SVM | 18.69 pp | 14.75 pp | 17.11 pp | 20.46 pp |
| DistilBERT | 36.03 pp | 35.92 pp | 35.92 pp | 36.81 pp |

*pp = percentage points.*

### Robustness Charts

The following charts are generated by `src/compare_all_models.py`:

- `results/all_models_deletion_comparison.png`
- `results/all_models_insertion_comparison.png`
- `results/all_models_substitution_comparison.png`
- `results/all_models_transposition_comparison.png`

### Key Findings

- DistilBERT achieved the highest clean-test accuracy among the three models in the current runs.
- Logistic Regression retained the highest accuracy at 20% corruption across all four evaluated strategies.
- Transposition produced the largest accuracy drop for each model at 20% corruption.
- The results indicate that clean-text accuracy alone is not sufficient to characterize robustness to character-level noise.
- In these experiments, the TF-IDF + Logistic Regression pipeline was more robust to the tested corruptions than the evaluated Linear SVM and fine-tuned DistilBERT models.

These findings are limited to the dataset, corruption procedures, training configurations, and evaluation runs used here. They should not be interpreted as universal claims about all models in these families.

**Reproducibility note:** Before treating the comparison as a final research conclusion, verify that all models are evaluated on identical corrupted test inputs and that the corruption rates, random seeds, and metric calculations are consistent. The separate DistilBERT evaluation currently reports a clean accuracy of 91.428%, while the rounded summary above displays 91.43%.

## Web Application

The project includes a Flask web application with three main pages.

### 1. Sentiment Analyzer

Enter a movie review and receive a predicted sentiment and model score.

### 2. Robustness Playground

Experiment with text corruption strategies and rates, then compare predictions on original and corrupted text.

### 3. Results Dashboard

Explore robustness experiment results and model performance comparisons.

The web application is intended for demonstration and research exploration; predictions are not guaranteed judgments of a review.

## Project Structure

```text
nlp-text-robustness/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── models/
├── notebooks/
├── results/
│   ├── baseline_metrics.json
│   ├── robustness_results.csv
│   ├── robustness_summary.json
│   ├── svm_metrics.json
│   ├── model_comparison_robustness.csv
│   ├── distilbert_metrics.json
│   ├── distilbert_robustness_results.csv
│   ├── distilbert_robustness_summary.json
│   └── all_models_*_comparison.png
├── src/
│   ├── corrupt_text.py
│   ├── train_baseline.py
│   ├── train_svm.py
│   ├── evaluate_robustness.py
│   ├── train_distilbert.py
│   ├── evaluate_distilbert_robustness.py
│   └── compare_all_models.py
├── templates/
│   ├── index.html
│   ├── playground.html
│   └── results.html
└── tests/
```

Some files may differ as the project evolves. Local datasets, trained model weights, and intermediate training checkpoints are excluded from Git.

## Installation and Setup

### Prerequisites

- Python 3.x
- Git
- A terminal or code editor
- Internet access for downloading the dataset and pretrained model

A CUDA-compatible GPU is recommended for DistilBERT training but is not required for the traditional machine learning experiments.

### 1. Clone the Repository

```bash
git clone https://github.com/Aayush-Thapa7/nlp-text-robustness.git
cd nlp-text-robustness
```

### 2. Create and Activate a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with your operating system, GPU, and CUDA requirements. Consult the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) if GPU acceleration is unavailable.

### 4. Train the Traditional Models

Train Logistic Regression:

```bash
python src/train_baseline.py
```

Train Linear SVM:

```bash
python src/train_svm.py
```

### 5. Train DistilBERT

```bash
python src/train_distilbert.py
```

This downloads the dataset and pretrained model if needed, then saves the fine-tuned model locally under `models/distilbert/`.

### 6. Evaluate Robustness

Run the traditional-model robustness evaluation:

```bash
python src/evaluate_robustness.py
```

Run the DistilBERT robustness evaluation:

```bash
python src/evaluate_distilbert_robustness.py
```

Create the combined comparison CSV and charts:

```bash
python src/compare_all_models.py
```

These experiments may take several minutes or longer depending on your hardware and whether required data and model files have already been downloaded.

### 7. Start the Web Application

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

Ensure the required local model files have been generated before using features that depend on them.

## Reproducibility

The project uses seeded corruption utilities and records evaluation metrics in JSON and CSV files. For repeatable comparisons, keep the following consistent:

- Dataset version and split
- Corruption implementation and rate
- Random seed and generated corrupted test inputs
- Model configuration and training seed
- Metric definitions and evaluation procedure
- Python and package versions

The combined comparison script consolidates existing result files; it does not retrain models or independently verify that separate experiments used identical corrupted inputs.

## Technologies

- Python
- Hugging Face Datasets and Transformers
- PyTorch
- scikit-learn
- TF-IDF
- Logistic Regression
- Linear SVM
- DistilBERT
- pandas and Matplotlib
- Flask
- HTML and CSS
- Git and GitHub

## Future Work

- Evaluate all models using exactly the same saved corrupted test sets.
- Investigate training-time noise augmentation and text normalization.
- Test additional corruption types and intensities.
- Report confidence intervals or repeated-seed results.
- Add automated tests for corruption functions and evaluation scripts.
- Improve the web interface and deployment workflow.

## Author

**Aayush Thapa**

GitHub: [@Aayush-Thapa7](https://github.com/Aayush-Thapa7)

## License

Check the repository for an existing `LICENSE` file before redistributing this project. If no license has been added, the project has not yet specified reuse permissions.
