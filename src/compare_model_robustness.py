
"""Compare Logistic Regression and Linear SVM under text corruption."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# Make imports work when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.corrupt_text import SUPPORTED_STRATEGIES, apply_corruption


# -------------------- Configuration --------------------

DATASET_NAME = "stanfordnlp/imdb"
RANDOM_SEED = 42
CORRUPTION_RATES = [0, 5, 10, 20]

MODEL_PATHS = {
    "Logistic Regression": PROJECT_ROOT / "models" / "sentiment_model.joblib",
    "Linear SVM": PROJECT_ROOT / "models" / "sentiment_svm.joblib",
}

CSV_PATH = PROJECT_ROOT / "results" / "model_comparison_robustness.csv"
JSON_PATH = PROJECT_ROOT / "results" / "model_comparison_robustness.json"


def main():
    print("=" * 68)
    print("MODEL ROBUSTNESS COMPARISON")
    print("Logistic Regression vs Linear SVM")
    print("=" * 68)

    # Check that both trained models exist.
    for model_name, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_name} model not found: {model_path}\n"
                "Please train both models before running this script."
            )

    # Load the existing models; this script does not retrain them.
    models = {}
    for model_name, model_path in MODEL_PATHS.items():
        print(f"\nLoading {model_name}...")
        models[model_name] = joblib.load(model_path)

    # Load the official IMDb test split.
    print("\nLoading IMDb test dataset...")
    dataset = load_dataset(DATASET_NAME)
    test_data = dataset["test"]

    test_texts = list(test_data["text"])
    test_labels = np.asarray(test_data["label"])

    print(f"Test reviews: {len(test_texts):,}")
    print(f"Corruption strategies: {', '.join(SUPPORTED_STRATEGIES)}")
    print(f"Corruption rates: {CORRUPTION_RATES}%")

    results = []
    clean_accuracies = {}

    # Evaluate each model on the unmodified test set first.
    print("\nCalculating clean-test performance...")
    for model_name, model in models.items():
        predictions = model.predict(test_texts)
        accuracy = float(accuracy_score(test_labels, predictions))
        macro_f1 = float(
            f1_score(test_labels, predictions, average="macro")
        )
        clean_accuracies[model_name] = accuracy

        print(
            f"{model_name}: accuracy={accuracy:.4f}, "
            f"macro-F1={macro_f1:.4f}"
        )

        results.append({
            "model": model_name,
            "strategy": "none",
            "rate_pct": 0,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "accuracy_drop_pct": 0.0,
        })

    # Each corrupted test set is generated only once, then used by
    # both models to ensure a fair, direct comparison.
    total_runs = len(SUPPORTED_STRATEGIES) * len(CORRUPTION_RATES)
    completed_runs = 0

    print("\nEvaluating corrupted reviews...")

    for strategy in SUPPORTED_STRATEGIES:
        for rate in CORRUPTION_RATES:
            if rate == 0:
                corrupted_texts = test_texts
            else:
                corrupted_texts = apply_corruption(
                    test_texts,
                    strategy=strategy,
                    rate=rate / 100.0,
                    seed=RANDOM_SEED,
                )


            for model_name, model in models.items():
                predictions = model.predict(corrupted_texts)

                accuracy = float(
                    accuracy_score(test_labels, predictions)
                )
                macro_f1 = float(
                    f1_score(test_labels, predictions, average="macro")
                )

                # Measure the accuracy drop relative to this model's
                # own clean-test accuracy, in percentage points.
                accuracy_drop_pct = (
                    clean_accuracies[model_name] - accuracy
                ) * 100

                results.append({
                    "model": model_name,
                    "strategy": strategy,
                    "rate_pct": rate,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "accuracy_drop_pct": accuracy_drop_pct,
                })

            completed_runs += 1
            print(
                f"[{completed_runs}/{total_runs}] "
                f"{strategy}, corruption={rate}% — complete"
            )

    # Save the detailed results as CSV.
    results_df = pd.DataFrame(results)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(CSV_PATH, index=False)

    # Save experiment metadata and results as JSON.
    report = {
        "experiment": "Logistic Regression vs Linear SVM robustness",
        "dataset": DATASET_NAME,
        "test_samples": len(test_texts),
        "random_seed": RANDOM_SEED,
        "corruption_strategies": list(SUPPORTED_STRATEGIES),
        "corruption_rates_pct": CORRUPTION_RATES,
        "model_paths": {
            name: str(path.relative_to(PROJECT_ROOT))
            for name, path in MODEL_PATHS.items()
        },
        "clean_test_accuracy": clean_accuracies,
        "results": results,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    # Print an easy-to-read summary.
    print("\n" + "=" * 68)
    print("CLEAN TEST ACCURACY")
    print("=" * 68)

    for model_name, accuracy in clean_accuracies.items():
        print(f"{model_name:22s}: {accuracy:.2%}")

    print("\n" + "=" * 68)
    print("ACCURACY AT 20% CORRUPTION")
    print("=" * 68)

    summary = results_df[results_df["rate_pct"] == 20]
    print(
        summary[
            ["model", "strategy", "accuracy", "accuracy_drop_pct"]
        ].to_string(index=False, formatters={
            "accuracy": "{:.2%}".format,
            "accuracy_drop_pct": "{:.2f}".format,
        })
    )

    print("\n" + "=" * 68)
    print("COMPARISON COMPLETE")
    print("=" * 68)
    print(f"CSV results:  {CSV_PATH}")
    print(f"JSON results: {JSON_PATH}")


if __name__ == "__main__":
    main()
