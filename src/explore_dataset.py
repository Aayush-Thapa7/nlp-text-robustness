"""
src/explore_dataset.py
======================
Milestone 2 — Dataset Exploration
----------------------------------
Loads the Stanford IMDb Large Movie Review dataset from the HuggingFace Hub
and prints a structured exploration report:

    1. Available splits and their sizes
    2. Feature names and label mapping
    3. Sample reviews from each class (truncated for readability)
    4. Class distribution counts
    5. Review-length statistics (characters and word counts)

This script does NOT train any model and does NOT save a copy of the dataset
to disk — it relies entirely on HuggingFace's built-in caching mechanism.

Usage
-----
From the project root (with the virtual environment active):

    python src/explore_dataset.py

Requirements
------------
    datasets>=2.14.0  (already listed in requirements.txt)
"""

# ── Standard-library imports ───────────────────────────────────────────────────
import random          # For reproducible sample selection
import sys             # For sys.exit on fatal errors
import textwrap        # For neatly wrapping long review text in the terminal

# ── Third-party imports ────────────────────────────────────────────────────────
# 'datasets' is the HuggingFace library that lets us download and work with
# benchmark datasets in a single line of code.
try:
    from datasets import load_dataset
except ImportError as e:
    print(
        "\n[ERROR] The 'datasets' library is not installed.\n"
        "        Run:  pip install datasets>=2.14.0\n"
        f"        Details: {e}"
    )
    sys.exit(1)

# ── Configuration ──────────────────────────────────────────────────────────────
# Fix the random seed so that every run picks the same sample reviews.
# Change this number if you want to see different examples.
RANDOM_SEED: int = 42

# How many sample reviews to display per class
SAMPLES_PER_CLASS: int = 2

# Maximum number of characters to print for each review preview
MAX_PREVIEW_CHARS: int = 400

# Terminal display width used for wrapping text
DISPLAY_WIDTH: int = 80

# Label mapping confirmed from the dataset card at
# https://huggingface.co/datasets/stanfordnlp/imdb
LABEL_MAP: dict[int, str] = {
    0: "Negative",
    1: "Positive",
}

# The IMDb dataset on HuggingFace Hub
DATASET_ID: str = "stanfordnlp/imdb"


# ── Helper functions ───────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    """Print a prominent section header to make the output easy to scan."""
    print()
    print("=" * DISPLAY_WIDTH)
    print(f"  {title}")
    print("=" * DISPLAY_WIDTH)


def truncate_text(text: str, max_chars: int) -> str:
    """
    Return the text trimmed to *max_chars* characters.

    If the text was shortened, an ellipsis (…) is appended so the reader
    knows the review continues beyond what is shown.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " …"


def wrap_text(text: str, indent: str = "    ") -> str:
    """
    Wrap *text* to DISPLAY_WIDTH columns with the given indentation.

    This keeps long reviews from overflowing off the right edge of narrow
    terminal windows.
    """
    return textwrap.fill(
        text,
        width=DISPLAY_WIDTH,
        initial_indent=indent,
        subsequent_indent=indent,
    )


def length_stats(values: list[int]) -> dict[str, float]:
    """
    Compute basic descriptive statistics for a list of integer lengths.

    Returns a dict with: min, max, mean, median, and std.
    We do the arithmetic ourselves so we don't need NumPy here.
    """
    n = len(values)
    if n == 0:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "std": 0.0}

    sorted_vals = sorted(values)
    mean = sum(sorted_vals) / n

    # Median: middle element (or average of two middle elements)
    mid = n // 2
    if n % 2 == 1:
        median = float(sorted_vals[mid])
    else:
        median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0

    # Population standard deviation
    variance = sum((v - mean) ** 2 for v in sorted_vals) / n
    std = variance ** 0.5

    return {
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
        "mean": mean,
        "median": median,
        "std": std,
    }


# ── Main exploration function ──────────────────────────────────────────────────

def explore_imdb() -> None:
    """
    Main entry point: load the IMDb dataset and print a full exploration report.
    """

    # ── 1. Load the dataset ────────────────────────────────────────────────────
    print_section("1 · Loading the IMDb Dataset")

    print(f"\n  Dataset ID : {DATASET_ID}")
    print(  "  Source     : https://huggingface.co/datasets/stanfordnlp/imdb")
    print(  "  Cache      : HuggingFace default (~/.cache/huggingface/datasets/)")
    print(  "  (No data is saved into the project repository.)\n")

    try:
        # load_dataset downloads the data on the first run and uses a local
        # cache on subsequent runs — so the download only happens once.
        dataset = load_dataset(DATASET_ID)
    except Exception as e:
        # Give the user a clear, actionable error message instead of a
        # cryptic Python traceback.
        print(
            f"\n[ERROR] Could not load the dataset '{DATASET_ID}'.\n"
            "\n  Possible causes:\n"
            "    • No internet connection on first run (cache not yet built)\n"
            "    • The dataset ID has changed on HuggingFace Hub\n"
            "    • The 'datasets' library version is too old\n"
            "\n  Try:\n"
            "    pip install --upgrade datasets\n"
            "\n  Or check the dataset page at:\n"
            "    https://huggingface.co/datasets/stanfordnlp/imdb\n"
            f"\n  Details: {e}"
        )
        sys.exit(1)

    print("  ✓ Dataset loaded successfully.")

    # ── 2. Splits and sizes ────────────────────────────────────────────────────
    print_section("2 · Available Splits and Their Sizes")

    print(f"\n  {'Split':<12}  {'Number of examples':>20}")
    print(f"  {'-'*12}  {'-'*20}")
    for split_name, split_data in dataset.items():
        print(f"  {split_name:<12}  {len(split_data):>20,}")

    print(
        "\n  Note: The IMDb dataset ships with 'train', 'test', and an\n"
        "  'unsupervised' split (unlabelled reviews, label = -1).\n"
        "  We focus on 'train' and 'test' for exploration."
    )

    # ── 3. Features and label mapping ─────────────────────────────────────────
    print_section("3 · Features and Label Mapping")

    # Use the training split as the reference — it always exists.
    train_split = dataset["train"]
    features = train_split.features

    print("\n  Feature columns:")
    for col_name, col_type in features.items():
        print(f"    • {col_name}: {col_type}")

    print("\n  Label mapping (verified against dataset metadata):")
    for label_int, label_str in LABEL_MAP.items():
        print(f"    {label_int}  →  {label_str}")

    # Cross-check: confirm the dataset's own label names match our mapping.
    dataset_label_names = features["label"].names
    print(f"\n  Dataset reports label names: {dataset_label_names}")
    print("  (Index 0 = first name = 'neg' = Negative, Index 1 = 'pos' = Positive)")

    # ── 4. Sample reviews from each class ─────────────────────────────────────
    print_section("4 · Sample Reviews (Truncated Previews)")

    # Seed the random number generator before any sampling.
    random.seed(RANDOM_SEED)
    print(f"\n  Random seed: {RANDOM_SEED}  |  Samples per class: {SAMPLES_PER_CLASS}")

    for label_int, label_str in LABEL_MAP.items():
        print(f"\n  ── {label_str.upper()} reviews (label = {label_int}) ──")

        # Collect the indices of all examples with this label, then sample.
        indices_for_label = [
            i for i, item in enumerate(train_split)
            if item["label"] == label_int
        ]
        chosen_indices = random.sample(indices_for_label, k=SAMPLES_PER_CLASS)

        for rank, idx in enumerate(chosen_indices, start=1):
            example = train_split[idx]
            preview = truncate_text(example["text"], MAX_PREVIEW_CHARS)

            print(f"\n  [{rank}] (train index {idx})")
            print(wrap_text(preview))

    # ── 5. Class counts and review-length statistics ───────────────────────────
    print_section("5 · Class Distribution and Review-Length Statistics")

    # We report stats for both the training and the test split.
    for split_name in ("train", "test"):
        split_data = dataset[split_name]
        print(f"\n  ── Split: '{split_name}' ({len(split_data):,} examples) ──")

        # --- Class counts ---
        counts: dict[int, int] = {k: 0 for k in LABEL_MAP}
        char_lengths: dict[int, list[int]] = {k: [] for k in LABEL_MAP}
        word_lengths: dict[int, list[int]] = {k: [] for k in LABEL_MAP}

        for item in split_data:
            lbl = item["label"]
            txt = item["text"]
            if lbl in counts:            # Skip unsupervised examples (label = -1)
                counts[lbl] += 1
                char_lengths[lbl].append(len(txt))
                word_lengths[lbl].append(len(txt.split()))

        total = sum(counts.values())
        print(f"\n  Class counts (total labelled = {total:,}):")
        print(f"  {'Class':<12}  {'Count':>8}  {'% of split':>12}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*12}")
        for lbl_int, lbl_str in LABEL_MAP.items():
            pct = 100.0 * counts[lbl_int] / total if total else 0.0
            print(f"  {lbl_str:<12}  {counts[lbl_int]:>8,}  {pct:>11.1f}%")

        # --- Length statistics ---
        print(f"\n  Review length statistics (by character count):")
        print(f"  {'Class':<12}  {'Min':>6}  {'Max':>7}  {'Mean':>8}  {'Median':>8}  {'Std':>8}")
        print(f"  {'-'*12}  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}")
        for lbl_int, lbl_str in LABEL_MAP.items():
            s = length_stats(char_lengths[lbl_int])
            print(
                f"  {lbl_str:<12}  {s['min']:>6,}  {s['max']:>7,}"
                f"  {s['mean']:>8.0f}  {s['median']:>8.0f}  {s['std']:>8.0f}"
            )

        print(f"\n  Review length statistics (by word count):")
        print(f"  {'Class':<12}  {'Min':>6}  {'Max':>7}  {'Mean':>8}  {'Median':>8}  {'Std':>8}")
        print(f"  {'-'*12}  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}")
        for lbl_int, lbl_str in LABEL_MAP.items():
            s = length_stats(word_lengths[lbl_int])
            print(
                f"  {lbl_str:<12}  {s['min']:>6,}  {s['max']:>7,}"
                f"  {s['mean']:>8.0f}  {s['median']:>8.0f}  {s['std']:>8.0f}"
            )

    # ── Summary ────────────────────────────────────────────────────────────────
    print_section("Exploration Complete")
    print(
        "\n  Key takeaways for this project:\n"
        "    • IMDb is a balanced binary dataset — equal positive/negative reviews.\n"
        "    • Reviews are full-length paragraphs, much longer than SST-2 sentences.\n"
        "    • The 'unsupervised' split contains 50,000 unlabelled reviews.\n"
        "    • Labels: 0 = Negative, 1 = Positive.\n"
        "\n  Next step → Milestone 3: text corruption and model evaluation.\n"
    )


# ── Script entry point ─────────────────────────────────────────────────────────
# This block only runs when the file is executed directly as a script,
# NOT when it is imported as a module by another file.
if __name__ == "__main__":
    explore_imdb()
