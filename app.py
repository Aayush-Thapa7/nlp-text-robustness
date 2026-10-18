
from pathlib import Path

import joblib
from flask import Flask, render_template, request
from src.corrupt_text import corrupt_text, SUPPORTED_STRATEGIES

# Project paths
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "sentiment_model.joblib"

app = Flask(__name__)

# Load the trained model
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    review = ""
    prediction = None
    probability = None

    if request.method == "POST":
        review = request.form.get("review", "").strip()

        if review:
            prediction_value = model.predict([review])[0]

            if str(prediction_value) == "1":
                prediction = "Positive"
            else:
                prediction = "Negative"

            # Get confidence scores when supported by the model
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([review])[0]
                classes = list(model.classes_)
                positive_index = classes.index(prediction_value)
                probability = float(probabilities[positive_index]) * 100

    return render_template(
        "index.html",
        review=review,
        prediction=prediction,
        probability=probability,
    )


@app.route("/playground", methods=["GET", "POST"])
def playground():
    original = ""
    corrupted = ""
    strategy = "substitution"
    rate = 0.10
    original_prediction = None
    corrupted_prediction = None
    original_score = None
    corrupted_score = None
    error = None

    if request.method == "POST":
        original = request.form.get("review", "").strip()
        strategy = request.form.get("strategy", "substitution")

        try:
            rate = float(request.form.get("rate", "10")) / 100

            if not original:
                raise ValueError("Please enter a movie review.")

            if strategy not in SUPPORTED_STRATEGIES:
                raise ValueError("Please select a valid corruption method.")

            if not 0 <= rate <= 0.20:
                raise ValueError("Choose a corruption rate between 0% and 20%.")

            corrupted = corrupt_text(
                original,
                strategy=strategy,
                rate=rate,
                seed=42,
            )

            original_prediction = str(model.predict([original])[0])
            corrupted_prediction = str(model.predict([corrupted])[0])

            # Convert numeric model labels into readable names.
            original_prediction = (
                "Positive" if original_prediction == "1"
                else "Negative" if original_prediction == "0"
                else original_prediction
            )
            corrupted_prediction = (
                "Positive" if corrupted_prediction == "1"
                else "Negative" if corrupted_prediction == "0"
                else corrupted_prediction
            )

            if hasattr(model, "predict_proba"):
                original_score = round(
                    max(model.predict_proba([original])[0]) * 100, 2
                )
                corrupted_score = round(
                    max(model.predict_proba([corrupted])[0]) * 100, 2
                )

        except ValueError as exc:
            error = str(exc)

    return render_template(
        "playground.html",
        original=original,
        corrupted=corrupted,
        strategy=strategy,
        rate=rate * 100,
        original_prediction=original_prediction,
        corrupted_prediction=corrupted_prediction,
        original_score=original_score,
        corrupted_score=corrupted_score,
        error=error,
        strategies=SUPPORTED_STRATEGIES,
    )
@app.route("/results")
def results():
    import json
    import pandas as pd

    results_path = ROOT_DIR / "results"
    csv_path = results_path / "robustness_results.csv"
    summary_path = results_path / "robustness_summary.json"

    with open(summary_path, "r", encoding="utf-8") as file:
        summary = json.load(file)

    data = pd.read_csv(csv_path)
    records = data.to_dict(orient="records")

    return render_template(
        "results.html",
        summary=summary,
        records=records
    )
if __name__ == "__main__":
    app.run(debug=True)
