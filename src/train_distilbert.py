
"""Fine-tune DistilBERT for IMDb sentiment classification."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_NAME = "stanfordnlp/imdb"
MODEL_NAME = "distilbert/distilbert-base-uncased"
RANDOM_SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 8
NUM_EPOCHS = 2
LEARNING_RATE = 2e-5

MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "distilbert"
RESULTS_PATH = PROJECT_ROOT / "results" / "distilbert_metrics.json"
TRAINING_OUTPUT_DIR = PROJECT_ROOT / "training_output" / "distilbert"


def main():
    print("=" * 68)
    print("DISTILBERT IMDb SENTIMENT TRAINING")
    print("=" * 68)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check your PyTorch GPU installation."
        )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    set_seed(RANDOM_SEED)

    print("\nLoading IMDb dataset...")
    dataset = load_dataset(DATASET_NAME)

    train_data = dataset["train"]
    test_data = dataset["test"]

    print(f"Training reviews: {len(train_data):,}")
    print(f"Test reviews: {len(test_data):,}")

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing IMDb reviews",
    )

    # Keep only the labeled training and test splits.
    tokenized_train = tokenized_dataset["train"]
    tokenized_test = tokenized_dataset["test"]

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("\nLoading pretrained DistilBERT...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(
                labels, predictions, average="macro"
            ),
        }

    # Transformers versions differ in some argument names.
    # Inspect the installed version to select the supported name.
    import inspect

    training_parameters = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    args = {
        "output_dir": str(TRAINING_OUTPUT_DIR),
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": 0.01,
        "logging_steps": 100,
        "save_strategy": "epoch",
        "fp16": True,
        "seed": RANDOM_SEED,
        "data_seed": RANDOM_SEED,
        "report_to": "none",
        "dataloader_num_workers": 0,
        "gradient_accumulation_steps": 1,
    }

    if "eval_strategy" in training_parameters:
        args["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in training_parameters:
        args["evaluation_strategy"] = "epoch"

    if "use_cpu" in training_parameters:
        args["use_cpu"] = False

    training_args = TrainingArguments(**args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nTraining configuration:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Maximum sequence length: {MAX_LENGTH}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print("  Mixed precision (FP16): enabled")
    print("\nTraining starts now. This may take a while.")

    start_time = time.time()
    trainer.train()
    training_seconds = time.time() - start_time

    print("\nEvaluating on the official IMDb test split...")
    evaluation = trainer.evaluate(eval_dataset=tokenized_test)

    # Save the model and tokenizer for later inference.
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))

    results = {
        "experiment": "DistilBERT IMDb sentiment classification",
        "dataset": DATASET_NAME,
        "pretrained_model": MODEL_NAME,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "random_seed": RANDOM_SEED,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "training_seconds": round(training_seconds, 2),
        "gpu": torch.cuda.get_device_name(0),
        "evaluation": {
            key: float(value)
            for key, value in evaluation.items()
            if isinstance(value, (int, float, np.number))
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\n" + "=" * 68)
    print("DISTILBERT TRAINING COMPLETE")
    print("=" * 68)
    print(f"Test accuracy: {evaluation.get('eval_accuracy', float('nan')):.4f}")
    print(f"Macro-F1:      {evaluation.get('eval_macro_f1', float('nan')):.4f}")
    print(f"Training time: {training_seconds / 60:.1f} minutes")
    print(f"Saved model:   {MODEL_OUTPUT_DIR}")
    print(f"Saved metrics: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
