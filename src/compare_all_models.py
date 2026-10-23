
    
  
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

traditional_path = RESULTS_DIR / "model_comparison_robustness.csv"
distilbert_path = RESULTS_DIR / "distilbert_robustness_results.csv"

if not traditional_path.exists():
    raise FileNotFoundError(f"Missing file: {traditional_path}")
if not distilbert_path.exists():
    raise FileNotFoundError(f"Missing file: {distilbert_path}")

# Load the existing Logistic Regression and SVM results.
traditional = pd.read_csv(traditional_path)
traditional = traditional.rename(columns={
    "rate_pct": "corruption_rate",
    "accuracy_drop_pct": "accuracy_drop_pp",
})
traditional = traditional[
    ["model", "strategy", "corruption_rate", "accuracy", "macro_f1",
     "accuracy_drop_pp"]
]

# Load DistilBERT results and standardize the model name/columns.
distilbert = pd.read_csv(distilbert_path).copy()
distilbert["model"] = "DistilBERT"
distilbert = distilbert.rename(columns={
    "corruption_rate": "corruption_rate",
})
distilbert = distilbert[
    ["model", "strategy", "corruption_rate", "accuracy", "macro_f1",
     "accuracy_drop_pp"]
]

combined = pd.concat([traditional, distilbert], ignore_index=True)
combined["corruption_rate"] = combined["corruption_rate"].astype(int)
combined["accuracy_percent"] = combined["accuracy"] * 100
combined["macro_f1_percent"] = combined["macro_f1"] * 100

# Sort rows for easier inspection.
model_order = ["Logistic Regression", "Linear SVM", "DistilBERT"]
combined["model"] = pd.Categorical(
    combined["model"], categories=model_order, ordered=True
)
combined = combined.sort_values(
    ["strategy", "corruption_rate", "model"]
).reset_index(drop=True)

csv_path = RESULTS_DIR / "all_models_robustness_comparison.csv"
combined.to_csv(csv_path, index=False)

# Create a chart for each corruption strategy. Each line shows one model.
strategies = ["deletion", "insertion", "substitution", "transposition"]
for strategy in strategies:
    subset = combined[
        (combined["strategy"] == strategy)
        & (combined["corruption_rate"] > 0)
    ]

    if subset.empty:
        print(f"Skipping {strategy}: no non-zero corruption rows found.")
        continue

    plt.figure(figsize=(8, 5))
    for model_name in model_order:
        model_data = subset[subset["model"] == model_name].sort_values(
            "corruption_rate"
        )
        if not model_data.empty:
            plt.plot(
                model_data["corruption_rate"],
                model_data["accuracy_percent"],
                marker="o",
                label=model_name,
            )

    plt.title(f"Model Robustness: {strategy.title()} Corruption")
    plt.xlabel("Character corruption rate (%)")
    plt.ylabel("Accuracy (%)")
    plt.xticks([5, 10, 20])
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    chart_path = RESULTS_DIR / f"all_models_{strategy}_comparison.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Saved chart: {chart_path}")

print("\nCombined comparison created.")
print(f"CSV: {csv_path}")
print("\nAccuracy at 20% corruption:")
twenty = combined[combined["corruption_rate"] == 20]
if not twenty.empty:
    print(
        twenty[["model", "strategy", "accuracy_percent",
                "accuracy_drop_pp"]].to_string(index=False)
    )
else:
    print("No 20% corruption rows were found.")

