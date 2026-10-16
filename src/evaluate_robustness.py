"""
src/evaluate_robustness.py
==========================
Milestone 4 -- Robustness Evaluation of the TF-IDF + Logistic Regression Model
--------------------------------------------------------------------------------
Evaluates the trained TF-IDF + Logistic Regression baseline against corrupted
versions of the IMDb test set.

What this script does
---------------------
  1. Re-trains the TF-IDF + Logistic Regression model on the same IMDb
     training split that was used for the original baseline (same seed,
     same hyperparameters) so predictions are identical.
  2. Loads the IMDb test split (25,000 reviews, labels unchanged).
  3. Applies each of the four character-level corruption strategies at each
     of four corruption rates: 0%, 5%, 10%, 20%.
  4. Reports accuracy and macro-F1 for every (strategy, rate) combination.
  5. At rate=0% the results should match the clean baseline (89.92% accuracy).
  6. Saves a structured CSV to results/robustness_results.csv.
  7. Saves an optional summary JSON to results/robustness_summary.json.

Why retrain instead of loading a saved model?
---------------------------------------------
  The baseline script does not save the trained pipeline to disk (saving a
  100k-feature TF-IDF + sklearn model as a pickle is large and excluded by
  .gitignore).  Re-training is fast (~10 s) and -- because we use the same
  seed and hyperparameters -- produces *identically* the same model and the
  same predictions on clean data.

Labels are NEVER corrupted
--------------------------
  Only the *text* of each test review is corrupted.  The ground-truth labels
  (0=Negative, 1=Positive) remain unchanged throughout all experiments.
  This is intentional: we want to measure how well the model still recognises
  the true sentiment when the text has noise in it.

CSV format (results/robustness_results.csv)
-------------------------------------------
  strategy, rate_pct, accuracy, macro_f1, accuracy_drop_pct
  deletion,      0,  0.8992,   0.8992,   0.00
  deletion,      5,  0.8851,   0.8851,   1.41
  ...

Usage
-----
From the project root (with .venv active):

    python src/evaluate_robustness.py

Requirements
------------
    datasets>=2.14.0   (HuggingFace dataset loader)
    scikit-learn>=1.3.0
    Both are already listed in requirements.txt.
"""

# ---- Standard-library imports -----------------------------------------------
import csv          # For writing the results CSV
import io           # For stdout UTF-8 reconfiguration (Windows fix)
import json         # For saving the summary JSON
import sys          # For sys.exit on fatal errors
import time         # For timing each step
from datetime import datetime  # For timestamping the output files
from pathlib import Path        # For creating the results/ directory safely

# ---- Windows terminal fix ---------------------------------------------------
# Windows terminals often default to cp1252, which cannot display many Unicode
# characters.  Switch stdout to UTF-8 before any print() calls.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

# ---- Third-party imports ----------------------------------------------------
try:
    from datasets import load_dataset
except ImportError:
    print(
        "\n[ERROR] The 'datasets' library is not installed.\n"
        "        Run:  pip install datasets>=2.14.0\n"
    )
    sys.exit(1)

try:
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
except ImportError:
    print(
        "\n[ERROR] scikit-learn is not installed.\n"
        "        Run:  pip install scikit-learn>=1.3.0\n"
    )
    sys.exit(1)

# ---- Local imports ----------------------------------------------------------
# We import from our own src package.  The script must be run from the project
# root (e.g., `python src/evaluate_robustness.py`) so that Python can find
# the `src` package.
try:
    from src.corrupt_text import apply_corruption, SUPPORTED_STRATEGIES
except ImportError:
    # Fallback: allow running with `python -m src.evaluate_robustness` or
    # if the user runs the script from inside the src/ directory.
    try:
        from corrupt_text import apply_corruption, SUPPORTED_STRATEGIES
    except ImportError as e:
        print(
            "\n[ERROR] Could not import from src/corrupt_text.py.\n"
            "        Make sure you run this script from the project root:\n"
            "            python src/evaluate_robustness.py\n"
            f"        Details: {e}\n"
        )
        sys.exit(1)


# ---- Configuration ----------------------------------------------------------

# The IMDb dataset on HuggingFace Hub -- same as used in train_baseline.py.
DATASET_ID: str = "stanfordnlp/imdb"

# Random seed -- must match the one used in train_baseline.py so the pipeline
# is identical and the 0%-corruption result matches the stored baseline.
RANDOM_SEED: int = 42

# ---- TF-IDF hyperparameters (identical to train_baseline.py) ----------------
NGRAM_RANGE: tuple[int, int] = (1, 2)
MAX_FEATURES: int = 100_000
SUBLINEAR_TF: bool = True

# ---- Logistic Regression hyperparameters (identical to train_baseline.py) ---
LR_C: float = 1.0
LR_MAX_ITER: int = 1000
LR_SOLVER: str = "lbfgs"

# ---- Corruption experiment settings -----------------------------------------
# Corruption rates expressed as percentages (0%, 5%, 10%, 20%).
# Internally converted to floats (0.0, 0.05, 0.10, 0.20) before calling
# the corruption functions.
CORRUPTION_RATES_PCT: list[int] = [0, 5, 10, 20]

# All four strategies (matches SUPPORTED_STRATEGIES in corrupt_text.py).
CORRUPTION_STRATEGIES: list[str] = list(SUPPORTED_STRATEGIES)

# Seed used for each corruption call.  Having a fixed seed ensures that
# running this script multiple times always produces the same corrupted texts.
CORRUPTION_SEED: int = 42

# ---- Clean baseline metrics (from results/baseline_metrics.json) ------------
# Stored here so we can compute the drop without re-loading the JSON file.
# These values must match the actual saved baseline metrics.
BASELINE_ACCURACY: float = 0.8992

# ---- Output paths -----------------------------------------------------------
RESULTS_DIR: Path = Path("results")
CSV_FILE: Path = RESULTS_DIR / "robustness_results.csv"
JSON_FILE: Path = RESULTS_DIR / "robustness_summary.json"

# ---- Display formatting -----------------------------------------------------
WIDTH: int = 72
LABEL_NAMES: list[str] = ["Negative", "Positive"]


# ---- Helper functions -------------------------------------------------------

def section(title: str) -> None:
    """Print a clearly visible section heading."""
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def step(msg: str) -> None:
    """Print a timestamped progress message."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}")


def build_pipeline() -> Pipeline:
    """
    Build a TF-IDF + Logistic Regression pipeline with exactly the same
    hyperparameters as in train_baseline.py.

    Returns
    -------
    Pipeline
        An unfitted sklearn Pipeline ready to be trained.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=NGRAM_RANGE,
                    max_features=MAX_FEATURES,
                    sublinear_tf=SUBLINEAR_TF,
                    strip_accents="unicode",
                    analyzer="word",
                    token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=LR_C,
                    max_iter=LR_MAX_ITER,
                    solver=LR_SOLVER,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_on_corrupted(
    pipeline: Pipeline,
    test_texts: list[str],
    test_labels: list[int],
    strategy: str,
    rate_pct: int,
    baseline_accuracy: float,
) -> dict:
    """
    Corrupt the test texts at a given rate, predict with the pipeline, and
    return a dict of metrics.

    Parameters
    ----------
    pipeline : Pipeline
        The already-fitted TF-IDF + LR pipeline.
    test_texts : list[str]
        Original (clean) test reviews.
    test_labels : list[int]
        Ground-truth labels -- NEVER corrupted.
    strategy : str
        Corruption strategy name.
    rate_pct : int
        Corruption rate as a percentage (0, 5, 10, or 20).

    Returns
    -------
    dict
        Keys: strategy, rate_pct, accuracy, macro_f1, accuracy_drop_pct.
    """
    rate = rate_pct / 100.0  # Convert percentage to float (e.g. 5 -> 0.05)

    if rate == 0.0:
        # At 0% corruption we skip the (expensive) corruption step and predict
        # on the clean texts directly.  This saves time and avoids any subtle
        # floating-point differences from a no-op corruption pass.
        corrupted_texts = test_texts
    else:
        # Apply the corruption strategy with a fixed seed for reproducibility.
        # Note: labels are NOT touched here -- only the text is corrupted.
        corrupted_texts = apply_corruption(
            test_texts,
            strategy=strategy,
            rate=rate,
            seed=CORRUPTION_SEED,
        )

    # Run the model's predict step.  The TF-IDF vectorizer was fit on
    # the clean training data; it will process corrupted texts through the
    # same vocabulary (unknown n-grams simply get zero weight).
    predictions = pipeline.predict(corrupted_texts)

    # Compute metrics
    acc = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average="macro")

    # Accuracy drop: how many percentage points below the clean baseline?
    # A positive drop means performance got worse.
    drop = (baseline_accuracy - acc) * 100.0

    return {
        "strategy":          strategy,
        "rate_pct":          rate_pct,
        "accuracy":          round(acc, 6),
        "macro_f1":          round(f1, 6),
        "accuracy_drop_pct": round(drop, 4),
    }


# ---- Main evaluation function -----------------------------------------------

def run_robustness_evaluation() -> None:
    """
    Main entry point: train the model, evaluate on all (strategy, rate) pairs,
    and save the results.
    """

    # ---- Step 1: Load the dataset -------------------------------------------
    section("Step 1 . Loading the IMDb Dataset")

    step(f"Requesting '{DATASET_ID}' from HuggingFace Hub ...")
    try:
        dataset = load_dataset(DATASET_ID)
    except Exception as e:
        print(
            f"\n[ERROR] Could not load '{DATASET_ID}'.\n"
            "  Possible causes:\n"
            "    * No internet connection on first run\n"
            "    * 'datasets' library too old -> pip install --upgrade datasets\n"
            f"  Details: {e}\n"
        )
        sys.exit(1)

    step("Dataset loaded successfully.")

    # Extract train and test texts / labels as plain Python lists.
    # This is the SAME split used in train_baseline.py -- we do not modify it.
    step("Extracting training split ...")
    train_texts: list[str] = dataset["train"]["text"]
    train_labels: list[int] = dataset["train"]["label"]

    step("Extracting test split ...")
    test_texts: list[str] = dataset["test"]["text"]
    test_labels: list[int] = dataset["test"]["label"]

    step(f"Training examples : {len(train_texts):,}")
    step(f"Test examples     : {len(test_texts):,}")

    # ---- Step 2: Train the pipeline -----------------------------------------
    section("Step 2 . Training the TF-IDF + Logistic Regression Pipeline")

    print(
        "\n  NOTE: We re-train with the IDENTICAL hyperparameters and seed as\n"
        "  train_baseline.py.  The resulting model is bit-for-bit the same,\n"
        "  so predictions on clean data will exactly match the stored baseline.\n"
    )

    pipeline = build_pipeline()

    step("Fitting pipeline on training data (this may take 1-3 minutes) ...")
    t0 = time.time()
    pipeline.fit(train_texts, train_labels)
    fit_time = time.time() - t0
    step(f"Training complete in {fit_time:.1f}s.")

    vocab_size = len(pipeline.named_steps["tfidf"].vocabulary_)
    step(f"TF-IDF vocabulary size: {vocab_size:,} n-grams.")

    # ---- Step 3: Sanity check at 0% corruption -------------------------------
    section("Step 3 . Sanity Check at 0% Corruption (Should Match Baseline)")

    clean_preds = pipeline.predict(test_texts)
    clean_acc = accuracy_score(test_labels, clean_preds)
    clean_f1  = f1_score(test_labels, clean_preds, average="macro")

    print(f"\n  Clean test accuracy : {clean_acc:.4f}  ({clean_acc*100:.2f}%)")
    print(f"  Stored baseline     : {BASELINE_ACCURACY:.4f}  ({BASELINE_ACCURACY*100:.2f}%)")
    print(f"  Clean macro-F1      : {clean_f1:.4f}  ({clean_f1*100:.2f}%)")

    # Check that our re-trained model matches the baseline to 4 decimal places.
    if abs(clean_acc - BASELINE_ACCURACY) > 0.0001:
        print(
            f"\n  [WARNING] The re-trained model accuracy ({clean_acc:.4f}) differs\n"
            f"  from the stored baseline ({BASELINE_ACCURACY:.4f}) by more than\n"
            "  0.01 percentage points.  The hyperparameters or dataset may have\n"
            "  changed.  Proceeding, but results may not be directly comparable.\n"
        )
    else:
        print(
            "\n  [OK] Re-trained model matches the stored baseline within tolerance.\n"
            "       All results at 0% corruption should be reported as the baseline."
        )

    # ---- Step 4: Run the full robustness grid --------------------------------
    section("Step 4 . Running Robustness Evaluation Grid")

    print(
        f"\n  Strategies : {CORRUPTION_STRATEGIES}\n"
        f"  Rates (%)  : {CORRUPTION_RATES_PCT}\n"
        f"  Total runs : {len(CORRUPTION_STRATEGIES) * len(CORRUPTION_RATES_PCT)}\n"
        f"  Seed       : {CORRUPTION_SEED}\n"
    )

    all_results: list[dict] = []

    for strategy in CORRUPTION_STRATEGIES:
        print(f"\n  -- Strategy: {strategy.upper()} --")
        for rate_pct in CORRUPTION_RATES_PCT:
            rate_label = f"{rate_pct:>3}%"
            step(f"  {strategy:15s} @ {rate_label} ...")

            t_start = time.time()
            result = evaluate_on_corrupted(
                pipeline,
                test_texts,
                test_labels,
                strategy,
                rate_pct,
                clean_acc,
)
            elapsed = time.time() - t_start

            # Print a one-line summary for this (strategy, rate) pair
            drop_sign = "+" if result["accuracy_drop_pct"] >= 0 else ""
            print(
                f"      Accuracy: {result['accuracy']*100:.2f}%  |  "
                f"Macro-F1: {result['macro_f1']*100:.2f}%  |  "
                f"Drop: {drop_sign}{result['accuracy_drop_pct']:.2f} pp  "
                f"({elapsed:.1f}s)"
            )
            all_results.append(result)

    # ---- Step 5: Display a formatted summary table ---------------------------
    section("Step 5 . Summary Table")

    # Column headers
    col_w = 15
    header = (
        f"  {'Strategy':<14}  {'Rate':>5}  "
        f"{'Accuracy':>9}  {'Macro-F1':>9}  {'Drop (pp)':>10}"
    )
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))

    prev_strategy = None
    for r in all_results:
        # Print a blank separator between strategy groups for readability
        if r["strategy"] != prev_strategy and prev_strategy is not None:
            print()
        prev_strategy = r["strategy"]

        drop_str = f"{r['accuracy_drop_pct']:+.2f}"
        print(
            f"  {r['strategy']:<14}  {r['rate_pct']:>4}%  "
            f"  {r['accuracy']*100:>7.2f}%  "
            f"  {r['macro_f1']*100:>7.2f}%  "
            f"  {drop_str:>9}"
        )

    # ---- Step 6: Save results to CSV ----------------------------------------
    section("Step 6 . Saving Results")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -- CSV --
    # Each row is one (strategy, rate) pair.
    # Column names are intentionally verbose for self-documentation.
    csv_fieldnames = [
        "strategy",
        "rate_pct",
        "accuracy",
        "macro_f1",
        "accuracy_drop_pct",
    ]
    try:
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        step(f"CSV saved  : {CSV_FILE.resolve()}")
    except OSError as e:
        print(f"\n  [WARNING] Could not save CSV: {e}")

    # -- JSON summary --
    # Stores everything needed to reproduce the experiment: hyperparameters,
    # dataset, corruption settings, and the full results table.
    summary = {
        "experiment": "Milestone 4 -- Robustness Evaluation (TF-IDF + LR)",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": DATASET_ID,
        "train_examples": len(train_texts),
        "test_examples": len(test_texts),
        "model": {
            "tfidf": {
                "ngram_range": list(NGRAM_RANGE),
                "max_features": MAX_FEATURES,
                "sublinear_tf": SUBLINEAR_TF,
                "actual_vocab_size": vocab_size,
            },
            "logistic_regression": {
                "C": LR_C,
                "solver": LR_SOLVER,
                "max_iter": LR_MAX_ITER,
                "random_state": RANDOM_SEED,
            },
        },
        "corruption": {
            "strategies": CORRUPTION_STRATEGIES,
            "rates_pct": CORRUPTION_RATES_PCT,
            "seed": CORRUPTION_SEED,
            "rate_definition": (
                "Per-character Bernoulli: each character is independently "
                "affected with probability = rate/100."
            ),
        },
        "baseline_accuracy": BASELINE_ACCURACY,
        "results": all_results,
        "timing_seconds": {
            "model_training": round(fit_time, 2),
        },
    }

    try:
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        step(f"JSON saved : {JSON_FILE.resolve()}")
    except OSError as e:
        print(f"\n  [WARNING] Could not save JSON: {e}")

    # ---- Final banner --------------------------------------------------------
    section("Done")
    print(
        f"\n  Robustness evaluation complete.\n"
        f"  Dataset      : {DATASET_ID}\n"
        f"  Test reviews : {len(test_texts):,}\n"
        f"  Strategies   : {', '.join(CORRUPTION_STRATEGIES)}\n"
        f"  Rates        : {CORRUPTION_RATES_PCT}\n"
        f"\n  Results CSV  : {CSV_FILE}\n"
        f"  Summary JSON : {JSON_FILE}\n"
        f"\n  Next step -> Milestone 5: visualise degradation curves and report.\n"
    )


# ---- Script entry point ------------------------------------------------------
if __name__ == "__main__":
    run_robustness_evaluation()
