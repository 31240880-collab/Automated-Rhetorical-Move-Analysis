import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression baseline.")
    parser.add_argument("--input", default=str(ROOT / "data/processed/cars_sentences.csv"))
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(
            f"Dataset not found: {path}\n"
            "Prepare a labelled CSV with columns 'sentence' and 'label'."
        )

    df = pd.read_csv(path)
    required = {"sentence", "label"}
    if not required.issubset(df.columns):
        raise SystemExit("CSV must contain columns named 'sentence' and 'label'.")
    df = df.dropna(subset=["sentence", "label"])
    df["sentence"] = df["sentence"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["sentence"] != "") & (df["label"] != "")]
    if df["label"].nunique() < 2:
        raise SystemExit("At least two distinct labels are needed to train a classifier.")

    # This row-level split is only a baseline starter. For research, split by paper ID.
    X_train, X_test, y_train, y_test = train_test_split(
        df["sentence"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    out_dir = ROOT / "outputs"
    model_dir = ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(out_dir / "baseline_classification_report.csv")
    labels = sorted(df["label"].unique())
    pd.DataFrame(
        confusion_matrix(y_test, predictions, labels=labels),
        index=labels, columns=labels
    ).to_csv(out_dir / "baseline_confusion_matrix.csv")
    joblib.dump(model, model_dir / "baseline_model.joblib")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Saved model: {model_dir / 'baseline_model.joblib'}")
    print(f"Saved metrics in: {out_dir}")

if __name__ == "__main__":
    main()
