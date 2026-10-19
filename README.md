# Robustness of Sentiment Classification Models Under Character-Level Text Corruption

A machine learning research project investigating how character-level text corruption affects sentiment classification accuracy.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Task](https://img.shields.io/badge/Task-Sentiment%20Analysis-purple.svg)]()

## Overview

Text classification systems often receive text containing typos, missing characters, inserted characters, or misspelled words. These changes can affect the predictions made by machine learning models.

This project evaluates the robustness of a sentiment classification model under controlled character-level corruption. It also provides a web application for testing sentiment predictions, experimenting with corrupted text, and viewing evaluation results.

**Research question:** How does character-level text corruption affect the accuracy of a sentiment classification model?

## Research Objectives

- Establish a baseline for sentiment classification on clean text.
- Evaluate the effects of different character-level corruption strategies.
- Measure model performance at different corruption intensities.
- Compare accuracy and macro-F1 across experiments.
- Provide an interactive web application for exploring model predictions.

## Dataset

The project uses the **IMDb movie review dataset** for binary sentiment classification.

| Property | Description |
|---|---|
| Dataset | IMDb movie reviews |
| Training examples | 25,000 |
| Test examples | 25,000 |
| Task | Binary sentiment classification |
| Classes | Positive and negative |

The dataset is used to train and evaluate the baseline model. Dataset files are not included in this repository.

## Model and Methodology

The baseline classifier uses a traditional machine learning pipeline:

1. **TF-IDF vectorization** converts text into numerical features using unigram and bigram features.
2. **Logistic Regression** predicts whether a review expresses positive or negative sentiment.
3. The trained model is evaluated on clean text and text modified using character-level corruption.

### Character-Level Corruption Strategies

| Strategy | Description |
|---|---|
| Deletion | Removes selected characters |
| Insertion | Adds characters at selected positions |
| Substitution | Replaces selected characters |
| Transposition | Swaps adjacent characters |

The experiments evaluate corruption rates of **0%, 5%, 10%, and 20%**. A rate of 0% represents clean text without corruption.

## Results

### Baseline Performance

| Metric | Score |
|---|---:|
| Test accuracy | 89.92% |
| Macro-F1 | 89.92% |

### Accuracy at 20% Corruption

| Corruption strategy | Accuracy | Accuracy drop |
|---|---:|---:|
| Deletion | 75.93% | 13.99 percentage points |
| Insertion | 78.00% | 11.92 percentage points |
| Substitution | 76.12% | 13.80 percentage points |
| Transposition | 73.88% | 16.04 percentage points |

Accuracy drop is measured relative to the clean baseline accuracy.

### Key Findings

- The model performs better on clean text than on heavily corrupted text.
- Insertion produces the highest accuracy among the four strategies at 20% corruption.
- Transposition produces the lowest accuracy and the largest accuracy drop at 20% corruption.
- Character-level changes can substantially affect sentiment classification performance.

These findings apply to the evaluated baseline model and experimental settings; they do not establish that every sentiment classifier behaves the same way.

## Web Application

The project includes a Flask web application with three main pages.

### 1. Sentiment Analyzer
Enter a movie review and receive a predicted sentiment and model confidence score.

### 2. Robustness Playground
Choose a corruption strategy and corruption rate to compare predictions for original and corrupted text.

### 3. Results Dashboard
Explore experiment results, accuracy comparisons, and evaluation metrics.

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
│   └── robustness_summary.json
├── src/
│   ├── corrupt_text.py
│   ├── train_baseline.py
│   └── ...
├── templates/
│   ├── index.html
│   ├── playground.html
│   └── results.html
└── tests/
```

Some generated files, including the trained model and local datasets, are excluded from Git. The exact files in each directory may change as the project develops.

## Installation and Setup

### Prerequisites

- Python installed on your computer
- Git
- A terminal or code editor such as VS Code

### 1. Clone the repository

```bash
git clone https://github.com/Aayush-Thapa7/nlp-text-robustness.git
cd nlp-text-robustness
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Train the baseline model

The trained model file is not committed to the repository. Generate it locally by running:

```bash
python src/train_baseline.py
```

Wait for training to finish and confirm that the model is saved in the `models/` directory.

### 5. Start the web application

```bash
python app.py
```

Open this address in your browser:

```text
http://127.0.0.1:5000
```

Use the navigation links to access the analyzer, robustness playground, and results dashboard.

## Reproducibility

The corruption utilities support seeded random behavior to help make experiments reproducible. Results may depend on the dataset version, dependency versions, model configuration, and experimental settings.

## Technologies

- Python
- scikit-learn
- TF-IDF
- Logistic Regression
- pandas
- Flask
- HTML and CSS
- Git and GitHub

## Future Work

- Evaluate additional classification models.
- Investigate more character-level corruption methods.
- Add further robustness metrics and visualizations.
- Expand automated testing.
- Improve the web interface and deployment workflow.

## Author

**Aayush Thapa**

GitHub: [@Aayush-Thapa7](https://github.com/Aayush-Thapa7)

## License

Check the repository for an existing `LICENSE` file before redistributing this project. If no license has been added, the project has not yet specified reuse permissions.
