
"""Train and evaluate a TF-IDF + Linear SVM sentiment classifier."""

import json
import time
from pathlib import Path
from datetime import datetime

import joblib
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Experiment settings: keep TF-IDF consistent with the baseline.
DATASET_ID = "stanfordnlp/imdb"
RANDOM_SEED = 42
NGRAM_RANGE = (1, 2)
MAX_FEATURES = 100_000
SUBLINEAR_TF = True

MODEL_PATH = MODELS_DIR / "sentiment_svm.joblib"
METRICS_PATH = RESULTS_DIR / "svm_metrics.json"


def main():
    print("=" * 60)
    print("IMDb Sentiment Classification: Linear SVM")
    print("=" * 60)

    # 1. Load the same official IMDb train/test splits.
    print("\n[1/5] Loading IMDb dataset...")
    dataset = load_dataset(DATASET_ID)

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    print(f"Training reviews: {len(train_texts):,}")
    print(f"Test reviews:     {len(test_texts):,}")

    # 2. Build the pipeline.
    # LinearSVC is a linear classifier suited to sparse TF-IDF features.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=NGRAM_RANGE,
            max_features=MAX_FEATURES,
            sublinear_tf=SUBLINEAR_TF,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",
            min_df=2,
        )),
        ("clf", LinearSVC(
            C=1.0,
            random_state=RANDOM_SEED,
            max_iter=2000,
        )),
    ])

    # 3. Train.
    print("\n[2/5] Training Linear SVM...")
    start = time.time()
    model.fit(train_texts, train_labels)
    fit_seconds = time.time() - start
    print(f"Training completed in {fit_seconds:.1f} seconds.")

    # 4. Evaluate on the official test split.
    print("\n[3/5] Evaluating on the test set...")
    start = time.time()
    predictions = model.predict(test_texts)
    predict_seconds = time.time() - start

    accuracy = accuracy_score(test_labels, predictions)
    macro_f1 = f1_score(test_labels, predictions, average="macro")
    report = classification_report(
        test_labels,
        predictions,
        target_names=["Negative", "Positive"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(test_labels, predictions)

    print(f"\nTest accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro-F1:      {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")
    print("\nClassification report:")
    print(classification_report(
        test_labels,
        predictions,
        target_names=["Negative", "Positive"],
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(cm)

    # 5. Save model and metrics separately from Logistic Regression.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "experiment": "TF-IDF + Linear SVM",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": DATASET_ID,
        "random_seed": RANDOM_SEED,
        "train_examples": len(train_texts),
        "test_examples": len(test_texts),
        "tfidf": {
            "ngram_range": list(NGRAM_RANGE),
            "max_features": MAX_FEATURES,
            "sublinear_tf": SUBLINEAR_TF,
            "actual_vocab_size": len(
                model.named_steps["tfidf"].vocabulary_
            ),
        },
        "linear_svm": {
            "C": 1.0,
            "max_iter": 2000,
        },
        "results": {
            "accuracy": round(float(accuracy), 6),
            "macro_f1": round(float(macro_f1), 6),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        },
        "timing_seconds": {
            "fit": round(fit_seconds, 2),
            "predict": round(predict_seconds, 2),
        },
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n[4/5] Saved SVM model to:")
    print(MODEL_PATH)
    print("\n[5/5] Saved SVM metrics to:")
    print(METRICS_PATH)
    print("\nLogistic Regression files have not been modified.")


if __name__ == "__main__":
    main()
