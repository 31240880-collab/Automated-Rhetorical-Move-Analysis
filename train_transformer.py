"""
Optional SciBERT fine-tuning starter.

Install optional dependencies:
    pip install torch transformers datasets scikit-learn

Expected CSV columns: sentence,label
Labels should be consistent, e.g. Move 1, Move 2, Move 3.
For a publishable evaluation, use paper-level splits instead of a random sentence split.
"""
import argparse
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "allenai/scibert_scivocab_uncased"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "data/processed/cars_sentences.csv"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--model-name", default=MODEL_NAME)
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        raise SystemExit(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path).dropna(subset=["sentence", "label"])
    if not {"sentence", "label"}.issubset(df.columns):
        raise SystemExit("CSV must contain 'sentence' and 'label' columns.")
    df["sentence"] = df["sentence"].astype(str)
    df["label"] = df["label"].astype(str)
    labels = sorted(df["label"].unique())
    if len(labels) < 2:
        raise SystemExit("At least two classes are required.")
    label2id = {label: idx for idx, label in enumerate(labels)}
    df["label_id"] = df["label"].map(label2id)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label_id"]
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=256)

    train_ds = Dataset.from_pandas(train_df[["sentence", "label_id"]].rename(columns={"label_id": "labels"}), preserve_index=False)
    test_ds = Dataset.from_pandas(test_df[["sentence", "label_id"]].rename(columns={"label_id": "labels"}), preserve_index=False)
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
    train_ds = train_ds.remove_columns(["sentence"])
    test_ds = test_ds.remove_columns(["sentence"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(labels), id2label={i: label for label, i in label2id.items()},
        label2id=label2id
    )
    training_args = TrainingArguments(
        output_dir=str(ROOT / "outputs/transformer_training"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=str(ROOT / "outputs/logs"),
        report_to="none"
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=test_ds)
    trainer.train()
    predictions = trainer.predict(test_ds)
    predicted_ids = predictions.predictions.argmax(axis=-1)
    true_ids = predictions.label_ids
    report = classification_report(
        true_ids, predicted_ids, labels=list(range(len(labels))),
        target_names=labels, output_dict=True, zero_division=0
    )

    output_dir = ROOT / "models/scibert_classifier"
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "label_mapping.json").write_text(json.dumps(label2id, indent=2), encoding="utf-8")
    (ROOT / "outputs/transformer_classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved Transformer model to {output_dir}")
    print(f"Saved classification report to {ROOT / 'outputs/transformer_classification_report.json'}")
    print("Caution: this starter uses a random sentence split. For research claims, implement paper-level split.")

if __name__ == "__main__":
    main()
