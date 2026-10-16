"""
src/train_baseline.py
=====================
Milestone 3 -- Baseline Classifier: TF-IDF + Logistic Regression
-----------------------------------------------------------------
Trains and evaluates a classical NLP pipeline on the IMDb sentiment
dataset, establishing the clean-data performance ceiling against which
later corruption experiments will be measured.

Pipeline:
    1. Load the official IMDb train / test splits from HuggingFace.
    2. Vectorize raw text with TF-IDF (unigrams + bigrams).
    3. Train a Logistic Regression classifier on the training split.
    4. Evaluate on the test split (accuracy, macro-F1, full report,
       confusion matrix).
    5. Save metrics to results/baseline_metrics.json.

Usage
-----
From the project root (with the virtual environment active):

    python src/train_baseline.py

Requirements
------------
    datasets>=2.14.0    (HuggingFace dataset loader)
    scikit-learn>=1.3.0 (TfidfVectorizer, LogisticRegression, metrics)
    Both are already listed in requirements.txt.

What this script does NOT do
-----------------------------
    * No text corruption or augmentation.
    * No model checkpoints saved to disk.
    * No dataset files saved to the repository.
    * No DistilBERT or any neural network.
    * No normalisation beyond what TF-IDF already handles.

Key concepts for beginners
---------------------------
TF-IDF (Term Frequency - Inverse Document Frequency):
    Converts a raw text string into a numeric vector.  Each dimension
    of the vector corresponds to a word (or word pair -- a bigram) in
    the vocabulary.  The value is high when a word is frequent in THIS
    document but rare across ALL documents, making it a strong signal
    for that particular review.  Common filler words like "the" get a
    near-zero weight automatically.

Logistic Regression:
    Despite its name, this is a classification algorithm, not a
    regression one.  It learns a weight for every TF-IDF feature and
    combines them into a score.  A sigmoid function then turns that
    score into a probability between 0 and 1 (negative vs positive).
    It is fast, interpretable, and works very well with high-dimensional
    sparse features like TF-IDF vectors.

Train / test separation:
    We train the model on the 25,000 TRAIN reviews and measure its
    performance on the 25,000 TEST reviews.  These sets never overlap,
    so the test score tells us how well the model generalises to
    reviews it has never seen before.  Using the test set during
    training (even just to tune hyperparameters) would give an
    artificially optimistic score -- a mistake called "data leakage".
"""

# ---- Standard-library imports -----------------------------------------------
import io       # For stdout UTF-8 reconfiguration
import json     # For saving metrics as a JSON file
import sys      # For sys.exit on fatal errors
import time     # For measuring how long each step takes
from datetime import datetime   # For timestamping the saved metrics file
from pathlib import Path        # For safely creating the results/ directory

# ---- Reconfigure stdout to UTF-8 --------------------------------------------
# Windows terminals often default to cp1252 encoding, which cannot display
# many Unicode characters.  Switching to UTF-8 prevents UnicodeEncodeError
# crashes.  This must happen before any print() calls.
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
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )
except ImportError:
    print(
        "\n[ERROR] scikit-learn is not installed.\n"
        "        Run:  pip install scikit-learn>=1.3.0\n"
    )
    sys.exit(1)

# ---- Configuration ----------------------------------------------------------

# Random seed used in LogisticRegression and anywhere else randomness appears.
# Keeping this fixed makes the experiment fully reproducible.
RANDOM_SEED: int = 42

# The IMDb dataset on HuggingFace Hub.
DATASET_ID: str = "stanfordnlp/imdb"

# ---- TF-IDF hyperparameters ----
# ngram_range=(1, 2): include both single words (unigrams) and adjacent word
# pairs (bigrams).  Bigrams capture short phrases like "not good" or "really
# great" that unigrams alone would miss.
NGRAM_RANGE: tuple[int, int] = (1, 2)

# max_features: keep only the N most frequent n-grams across the training
# corpus.  This caps memory usage; 100,000 features is a good balance between
# coverage and RAM on a typical laptop.
MAX_FEATURES: int = 100_000

# sublinear_tf=True: apply log(1 + tf) instead of raw term frequency.
# This dampens the effect of very frequent words (e.g., "movie" appears
# 50 times vs 5 times -- sublinear scaling makes the difference smaller).
SUBLINEAR_TF: bool = True

# ---- Logistic Regression hyperparameters ----
# C: inverse regularisation strength.  Smaller C = more regularisation
# (penalises large weights more heavily, reducing overfitting).
# C=1.0 is the scikit-learn default and works well as a starting point.
LR_C: float = 1.0

# max_iter: maximum number of solver iterations.  IMDb has 25,000 long
# reviews and 100k features, so the solver needs more iterations than
# the default (100) to fully converge.
LR_MAX_ITER: int = 1000

# solver: the optimisation algorithm used internally.
# 'lbfgs' is a memory-efficient quasi-Newton method, well-suited to
# high-dimensional text classification problems.
LR_SOLVER: str = "lbfgs"

# ---- Output path ----
# The metrics JSON is saved here.  The results/ directory is created
# automatically if it does not already exist.
RESULTS_DIR: Path = Path("results")
METRICS_FILE: Path = RESULTS_DIR / "baseline_metrics.json"

# ---- Label names (for display) ----
LABEL_NAMES: list[str] = ["Negative", "Positive"]

# ---- Display width -----------------------------------------------------------
WIDTH: int = 72


# ---- Helper functions -------------------------------------------------------

def section(title: str) -> None:
    """Print a clearly visible section heading."""
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def step(message: str) -> None:
    """Print a single progress step with a timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {message}")


# ---- Main function ----------------------------------------------------------

def train_and_evaluate() -> None:
    """
    Full training and evaluation pipeline.

    Steps
    -----
    1. Load dataset
    2. Extract texts and labels from each split
    3. Build and fit the TF-IDF + LR pipeline on the train split
    4. Predict on the test split
    5. Compute and display metrics
    6. Save metrics to JSON
    """

    # ---- Step 1: Load the dataset -------------------------------------------
    section("Step 1 . Loading the IMDb Dataset")

    step(f"Requesting '{DATASET_ID}' from HuggingFace Hub ...")
    t0 = time.time()

    try:
        dataset = load_dataset(DATASET_ID)
    except Exception as e:
        print(
            f"\n[ERROR] Could not load '{DATASET_ID}'.\n"
            "\n  Possible causes:\n"
            "    * No internet connection on first run\n"
            "    * Dataset ID changed on HuggingFace Hub\n"
            "    * 'datasets' library too old  ->  pip install --upgrade datasets\n"
            f"\n  Details: {e}\n"
        )
        sys.exit(1)

    elapsed = time.time() - t0
    step(f"Dataset loaded in {elapsed:.1f}s.")
    step(f"Splits available: {list(dataset.keys())}")

    # ---- Step 2: Extract texts and labels -----------------------------------
    section("Step 2 . Extracting Texts and Labels")

    # The IMDb train and test splits each have 25,000 labelled reviews.
    # We convert them to plain Python lists because scikit-learn expects
    # ordinary iterables, not HuggingFace Dataset objects.
    step("Extracting training split ...")
    train_texts: list[str] = dataset["train"]["text"]
    train_labels: list[int] = dataset["train"]["label"]

    step("Extracting test split ...")
    test_texts: list[str] = dataset["test"]["text"]
    test_labels: list[int] = dataset["test"]["label"]

    step(f"Training examples : {len(train_texts):,}")
    step(f"Test examples     : {len(test_texts):,}")

    # Quick sanity check: ensure no unexpected label values slipped through.
    unique_train_labels = set(train_labels)
    unique_test_labels = set(test_labels)
    if unique_train_labels != {0, 1} or unique_test_labels != {0, 1}:
        print(
            f"\n[WARNING] Unexpected label values detected.\n"
            f"  Train labels: {unique_train_labels}\n"
            f"  Test labels:  {unique_test_labels}\n"
            "  Expected only 0 (Negative) and 1 (Positive).\n"
            "  Proceeding, but results may be unreliable.\n"
        )

    # ---- Step 3: Build and train the pipeline --------------------------------
    section("Step 3 . Building and Training the TF-IDF + LR Pipeline")

    # A scikit-learn Pipeline chains steps so that:
    #   a) The vectorizer is fit on training text only (no test leakage).
    #   b) The same transformation is automatically applied to test text.
    #   c) Both steps are trained / used together with a single .fit() call.
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=NGRAM_RANGE,   # unigrams + bigrams
                    max_features=MAX_FEATURES, # vocabulary size cap
                    sublinear_tf=SUBLINEAR_TF, # log-scale term frequencies
                    strip_accents="unicode",   # normalise accented characters
                    analyzer="word",           # tokenise at word level
                    token_pattern=r"\b[a-zA-Z][a-zA-Z]+\b",  # letters only
                    min_df=2,                  # ignore terms seen only once
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=LR_C,
                    max_iter=LR_MAX_ITER,
                    solver=LR_SOLVER,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,   # use all available CPU cores
                ),
            ),
        ]
    )

    step("Pipeline configuration:")
    print(f"\n    TF-IDF:")
    print(f"      ngram_range  = {NGRAM_RANGE}  (unigrams + bigrams)")
    print(f"      max_features = {MAX_FEATURES:,}")
    print(f"      sublinear_tf = {SUBLINEAR_TF}")
    print(f"      min_df       = 2  (drop hapax legomena)")
    print(f"\n    Logistic Regression:")
    print(f"      C            = {LR_C}  (regularisation inverse strength)")
    print(f"      solver       = {LR_SOLVER!r}")
    print(f"      max_iter     = {LR_MAX_ITER}")
    print(f"      random_state = {RANDOM_SEED}")

    step("Fitting on training data (this may take 1-3 minutes) ...")
    t_fit_start = time.time()

    # .fit() on the Pipeline does two things in order:
    #   1. Calls tfidf.fit_transform(train_texts) -- builds the vocabulary
    #      from training texts and converts them to a sparse TF-IDF matrix.
    #   2. Calls clf.fit(X_tfidf, train_labels) -- trains the classifier.
    # Neither step ever sees the test texts at this point.
    pipeline.fit(train_texts, train_labels)

    fit_seconds = time.time() - t_fit_start
    step(f"Training complete in {fit_seconds:.1f}s.")

    # Report the actual vocabulary size (may be < max_features if the
    # corpus has fewer unique n-grams than the cap).
    vocab_size = len(pipeline.named_steps["tfidf"].vocabulary_)
    step(f"TF-IDF vocabulary size: {vocab_size:,} n-grams.")

    # ---- Step 4: Evaluate on the test split ----------------------------------
    section("Step 4 . Evaluating on the Test Split")

    step("Generating predictions on test data ...")
    t_pred_start = time.time()

    # .predict() on the Pipeline applies the SAME tfidf transformation
    # (using the vocabulary built from training data only) to test_texts,
    # then runs the classifier.
    predictions: list[int] = pipeline.predict(test_texts)

    pred_seconds = time.time() - t_pred_start
    step(f"Predictions generated in {pred_seconds:.1f}s.")

    # ---- Step 5: Compute and display metrics --------------------------------
    section("Step 5 . Metrics")

    # -- Accuracy --
    # Fraction of test examples classified correctly.
    # For a balanced dataset (12,500 pos + 12,500 neg), accuracy is a fair
    # summary metric.  50% would be random-chance level.
    accuracy = accuracy_score(test_labels, predictions)

    # -- Macro-F1 --
    # F1 = harmonic mean of precision and recall.
    # "Macro" means: compute F1 for each class separately, then average them.
    # This treats both classes equally regardless of their sizes.
    macro_f1 = f1_score(test_labels, predictions, average="macro")

    # -- Full classification report --
    # Precision, recall, and F1 for each individual class, plus weighted and
    # macro averages.  target_names replaces 0/1 with readable class labels.
    report = classification_report(
        test_labels,
        predictions,
        target_names=LABEL_NAMES,
    )

    # -- Confusion matrix --
    # A 2x2 table:
    #   [ TN  FP ]     TN = True Negative  (correctly called Negative)
    #   [ FN  TP ]     TP = True Positive  (correctly called Positive)
    #                  FP = False Positive (called Positive, actually Negative)
    #                  FN = False Negative (called Negative, actually Positive)
    cm = confusion_matrix(test_labels, predictions)

    # Display accuracy and F1
    print(f"\n  Test Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Macro-F1 Score : {macro_f1:.4f}  ({macro_f1*100:.2f}%)")

    # Display classification report
    print(f"\n  Classification Report:\n")
    for line in report.splitlines():
        print(f"    {line}")

    # Display confusion matrix
    print(f"\n  Confusion Matrix (rows = actual, cols = predicted):\n")
    print(f"    {'':15s}  {'Pred: Negative':>16}  {'Pred: Positive':>15}")
    print(f"    {'-'*50}")
    row_labels = ["Actual: Negative", "Actual: Positive"]
    for row_label, row in zip(row_labels, cm):
        vals = "  ".join(f"{v:>14,}" for v in row)
        print(f"    {row_label:<16}  {vals}")

    # ---- Step 6: Save metrics to JSON ----------------------------------------
    section("Step 6 . Saving Metrics")

    # Build the metrics dictionary.
    # We store everything needed to reproduce and interpret the result:
    # the hyperparameters, the dataset, the timestamp, and the numbers.
    metrics: dict = {
        "experiment": "Milestone 3 -- TF-IDF + Logistic Regression Baseline",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": DATASET_ID,
        "random_seed": RANDOM_SEED,
        "train_examples": len(train_texts),
        "test_examples": len(test_texts),
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
        },
        "results": {
            "accuracy": round(accuracy, 6),
            "macro_f1": round(macro_f1, 6),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        },
        "timing_seconds": {
            "fit": round(fit_seconds, 2),
            "predict": round(pred_seconds, 2),
        },
    }

    # Create the results directory if it does not already exist.
    # exist_ok=True means no error is raised if the directory is already there.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            # indent=2 makes the JSON human-readable in any text editor.
            json.dump(metrics, f, indent=2)
        step(f"Metrics saved to: {METRICS_FILE.resolve()}")
    except OSError as e:
        print(
            f"\n[WARNING] Could not save metrics to '{METRICS_FILE}'.\n"
            f"          Details: {e}\n"
            "          The metrics were printed above and are not lost.\n"
        )

    # Note about .gitignore: the project .gitignore excludes *.json by default.
    # This means baseline_metrics.json will NOT be committed to Git unless you
    # explicitly track it with:
    #     git add -f results/baseline_metrics.json
    # That is intentional -- large prediction files should stay local.
    # For this small metrics file you may choose to track it.

    # ---- Summary banner ------------------------------------------------------
    section("Done")
    print(
        f"\n  Baseline model (TF-IDF + Logistic Regression)\n"
        f"  Dataset      : {DATASET_ID}\n"
        f"  Train size   : {len(train_texts):,} reviews\n"
        f"  Test size    : {len(test_texts):,} reviews\n"
        f"  Vocabulary   : {vocab_size:,} n-grams\n"
        f"\n  Test Accuracy  : {accuracy*100:.2f}%\n"
        f"  Macro-F1 Score : {macro_f1*100:.2f}%\n"
        f"\n  Metrics file : {METRICS_FILE}\n"
        f"\n  Next step -> Milestone 4: apply text corruptions and re-evaluate.\n"
    )


# ---- Script entry point ------------------------------------------------------
# This block runs ONLY when the file is executed directly as a script.
# Importing the module from another file will NOT trigger training.
if __name__ == "__main__":
    train_and_evaluate()
