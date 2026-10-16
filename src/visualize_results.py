
"""Visualize sentiment-model robustness under character-level corruption."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
CSV_PATH = RESULTS_DIR / "robustness_results.csv"


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found: {CSV_PATH}\n"
            "Run src/evaluate_robustness.py first."
        )

    df = pd.read_csv(CSV_PATH)

    required_columns = {"strategy", "rate_pct", "accuracy"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing expected CSV columns: {sorted(missing)}\n"
            f"Found columns: {list(df.columns)}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Graph 1: Accuracy degradation curves
    plt.figure(figsize=(9, 6))

    for strategy, group in df.groupby("strategy"):
        group = group.sort_values("rate_pct")
        plt.plot(
            group["rate_pct"],
            group["accuracy"] * 100,
            marker="o",
            linewidth=2,
            label=strategy.capitalize(),
        )

    plt.title("Sentiment Classification Robustness")
    plt.xlabel("Character Corruption Rate (%)")
    plt.ylabel("Test Accuracy (%)")
    plt.xticks(sorted(df["rate_pct"].unique()))
    plt.grid(True, alpha=0.3)
    plt.legend(title="Corruption Strategy")
    plt.tight_layout()

    curve_path = RESULTS_DIR / "accuracy_degradation.png"
    plt.savefig(curve_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Graph 2: Accuracy drop at each corruption rate
    pivot = df.pivot(
        index="rate_pct",
        columns="strategy",
        values="accuracy",
    ).sort_index()

    baseline = pivot.loc[0]
    drops = (baseline - pivot) * 100

    plt.figure(figsize=(9, 6))

    for strategy in drops.columns:
        plt.plot(
            drops.index,
            drops[strategy],
            marker="o",
            linewidth=2,
            label=strategy.capitalize(),
        )

    plt.title("Accuracy Drop Under Character-Level Corruption")
    plt.xlabel("Character Corruption Rate (%)")
    plt.ylabel("Accuracy Drop (percentage points)")
    plt.xticks(sorted(drops.index.unique()))
    plt.axhline(0, linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Corruption Strategy")
    plt.tight_layout()

    drop_path = RESULTS_DIR / "accuracy_drop.png"
    plt.savefig(drop_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nVisualization complete!")
    print(f"Accuracy curves: {curve_path}")
    print(f"Accuracy drop:   {drop_path}")


if __name__ == "__main__":
    main()
