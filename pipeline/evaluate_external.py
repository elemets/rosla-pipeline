"""Evaluate BERT-only (before) vs BERT+regex (after) on the external updated dataset.

Reproduces the evaluate.py external baseline (best-threshold BERT predictions,
per-class metrics) then re-scores after the NFLIS regex supplement/correction
pass from regex_classifier.py, printing a before/after comparison table.

Usage:
    python evaluate_external.py
    python evaluate_external.py --input ../data/external_updated.xlsx --model bioclinicalbert
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    hamming_loss,
)

from classify import TextDataset, apply_regex_classifier, create_text_col

ROOT = Path(__file__).resolve().parent.parent

DRUG_COLS = [
    "Methamphetamine",
    "Heroin",
    "Cocaine",
    "Fentanyl",
    "Alcohol",
    "Prescription.opioids",
    "Any Opioids",
    "Benzodiazepines",
    "Others",
    "Any Drugs",
]

LABEL_RENAMES = {
    "Any opioid": "Any Opioids",
    "Prescription opioids": "Prescription.opioids",
}


def load_external(input_path: Path):
    try:
        df = pd.read_excel(input_path).rename(columns=LABEL_RENAMES)
    except:
        df = pd.read_csv(input_path).rename(columns=LABEL_RENAMES)
    if "Any Drugs" not in df.columns:
        old = pd.read_csv(
            ROOT / "data" / "external_test.csv",
            usecols=["Case.Number", "Any Drugs"],
        ).drop_duplicates("Case.Number")
        df = df.merge(old, on="Case.Number", how="left")
        df["Any Drugs"] = df["Any Drugs"].fillna(0).astype(int)
    y_true = df[DRUG_COLS].astype(int).reset_index(drop=True)
    feats = df.drop(columns=DRUG_COLS).rename(columns={"text": "CauseA"})
    feats = create_text_col(feats).reset_index(drop=True)
    return feats, y_true


def predict_probs(texts, model_dir: Path, batch_size: int = 32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=len(DRUG_COLS),
        problem_type="multi_label_classification",
    ).to(device)
    model.eval()
    loader = DataLoader(TextDataset(texts, tokenizer), batch_size=batch_size)
    probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            probs.append(torch.sigmoid(model(**inputs).logits).cpu())
    return torch.cat(probs).numpy()


def per_class_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    rows = {
        col: {
            "accuracy": accuracy_score(y_true[col], y_pred[col]),
            "precision": precision_score(y_true[col], y_pred[col], zero_division=0),
            "recall": recall_score(y_true[col], y_pred[col], zero_division=0),
            "f1": f1_score(y_true[col], y_pred[col], zero_division=0),
        }
        for col in DRUG_COLS
    }
    return pd.DataFrame(rows).T


def macro_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame, probs=None) -> dict:
    metrics = {
        "Hamming Loss": hamming_loss(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro F1 Score": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if probs is not None:
        metrics["Macro AUC ROC"] = roc_auc_score(y_true, probs, average="macro")
    return metrics


def print_macro(title: str, metrics: dict):
    print(f"\n=== {title} ===")
    for name, value in metrics.items():
        print(f"{name}: {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "data" / "recoded_ext_test_v2.csv"))
    parser.add_argument("--model", default="bioclinicalbert")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "eval_external_regex_comparison_fabrice.csv"),
    )
    args = parser.parse_args()

    feats, y_true = load_external(Path(args.input))
    model_dir = ROOT / "models" / "bert_models" / args.model

    probs = predict_probs(feats["text"].tolist(), model_dir, args.batch_size)
    with open(model_dir / "best_thresholds.json") as f:
        thresholds = json.load(f)

    before = pd.DataFrame(
        {col: (probs[:, i] >= thresholds[col]).astype(int) for i, col in enumerate(DRUG_COLS)}
    )
    pred_df = pd.concat([feats, before], axis=1)

    after_df = apply_regex_classifier(pred_df)
    after = after_df[DRUG_COLS].astype(int)

    m_before = per_class_metrics(y_true, before)
    m_after = per_class_metrics(y_true, after)
    comparison = m_before.join(m_after, lsuffix="_before", rsuffix="_after")
    comparison["accuracy_delta"] = comparison["accuracy_after"] - comparison["accuracy_before"]
    comparison["f1_delta"] = comparison["f1_after"] - comparison["f1_before"]

    print_macro("BERT only (before regex) - macro metrics", macro_metrics(y_true, before, probs))
    print_macro("BERT + regex (after) - macro metrics", macro_metrics(y_true, after))

    print("\n=== BERT only (before regex) ===")
    print(m_before.round(4))
    print("\n=== BERT + regex (after) ===")
    print(m_after.round(4))
    print("\n=== Delta (after - before) ===")
    print(comparison[["accuracy_delta", "f1_delta"]].round(4))

    comparison.to_csv(args.output)
    print(f"\nSaved comparison -> {args.output}")


if __name__ == "__main__":
    main()
