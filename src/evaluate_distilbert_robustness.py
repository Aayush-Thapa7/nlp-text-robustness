
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from corrupt_text import apply_corruption


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "distilbert"
RESULTS_DIR = PROJECT_ROOT / "results"

MAX_LENGTH = 256
BATCH_SIZE = 16
SEED = 42

CORRUPTION_STRATEGIES = [
    "deletion",
    "insertion",
    "substitution",
    "transposition",
]
CORRUPTION_RATES = [0, 5, 10, 20]


class ReviewDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, index):
        return {
            key: torch.tensor(value[index], dtype=torch.long)
            for key, value in self.encodings.items()
        }


def predict(texts, tokenizer, model, device):
    dataset = ReviewDataset(texts, tokenizer)

    def collate_fn(batch):
        return tokenizer.pad(
            batch,
            padding=True,
            return_tensors="pt",
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    predictions = []

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predictions.extend(
                torch.argmax(logits, dim=1).cpu().tolist()
            )

    return np.array(predictions)


def main():
    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"DistilBERT model not found at {MODEL_PATH}. "
            "Run python src/train_distilbert.py first."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH)
    ).to(device)

    print("Loading IMDb test dataset...")
    dataset = load_dataset("stanfordnlp/imdb")
    test_data = dataset["test"]

    texts = list(test_data["text"])
    labels = np.array(test_data["label"])

    print(f"Test reviews: {len(texts):,}")

    print("\nEvaluating clean test reviews...")
    clean_predictions = predict(texts, tokenizer, model, device)
    clean_accuracy = accuracy_score(labels, clean_predictions)
    clean_f1 = f1_score(
        labels, clean_predictions, average="macro"
    )

    print(f"Clean accuracy: {clean_accuracy:.4f}")
    print(f"Clean macro-F1: {clean_f1:.4f}")

    rows = []

    # Include clean performance once as the 0% reference.
    rows.append({
        "strategy": "none",
        "corruption_rate": 0,
        "accuracy": clean_accuracy,
        "macro_f1": clean_f1,
        "accuracy_drop_pp": 0.0,
    })

    for strategy in CORRUPTION_STRATEGIES:
        for rate in CORRUPTION_RATES:
            if rate == 0:
                continue

            print(f"\nTesting {strategy} at {rate}% corruption...")

            corrupted_texts = apply_corruption(
                texts,
                strategy=strategy,
                rate=rate / 100.0,
                seed=SEED,
            )

            predictions = predict(
                corrupted_texts, tokenizer, model, device
            )

            accuracy = accuracy_score(labels, predictions)
            macro_f1 = f1_score(
                labels, predictions, average="macro"
            )

            accuracy_drop_pp = (
                clean_accuracy - accuracy
            ) * 100

            rows.append({
                "strategy": strategy,
                "corruption_rate": rate,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "accuracy_drop_pp": accuracy_drop_pp,
            })

            print(
                f"Accuracy: {accuracy:.4f} | "
                f"Macro-F1: {macro_f1:.4f} | "
                f"Accuracy drop: {accuracy_drop_pp:.2f} pp"
            )

    results = pd.DataFrame(rows)

    csv_path = RESULTS_DIR / "distilbert_robustness_results.csv"
    json_path = RESULTS_DIR / "distilbert_robustness_summary.json"

    results.to_csv(csv_path, index=False)

    summary = {
        "model": "DistilBERT",
        "dataset": "stanfordnlp/imdb",
        "clean_accuracy": float(clean_accuracy),
        "clean_macro_f1": float(clean_f1),
        "test_samples": len(texts),
        "corruption_strategies": CORRUPTION_STRATEGIES,
        "corruption_rates_percent": CORRUPTION_RATES,
        "results": rows,
        "evaluation_time_minutes": round(
            (time.time() - start_time) / 60, 2
        ),
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nEvaluation complete!")
    print(f"CSV saved to: {csv_path}")
    print(f"JSON saved to: {json_path}")
    print(f"Total time: {summary['evaluation_time_minutes']} minutes")


if __name__ == "__main__":
    main()
