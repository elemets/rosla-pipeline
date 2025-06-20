import typer
import pandas as pd
from datasets import Dataset
from rapid_overdose_classification.bert_modeling.constants import (
    device,
    drug_cols,
    MODEL_PATH,
    REPORTS_OUTPUT_PATH,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import numpy as np
import torch
from datetime import datetime
import json
import time

app = typer.Typer()


def predict_bert(input_data: str, model_type: str, text_col: str):
    """
    Predicts labels for a given dataset using a BERT model.

    Args:
        input_data (str): Path to the CSV file containing the input data.
        model_type (str): Type of the BERT model to be used for prediction.
        text_col (str): Name of the column containing text data in the input CSV file.

    Returns:
        None
    """
    # Load data to do the prediction on
    pred_df = pd.read_csv(input_data)
    texts = pred_df[text_col].tolist()
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}{model_type}")

    # Process texts in batches to avoid overflow
    all_probs = []
    batch_size = 8  # Adjust based on memory constraints

    model = AutoModelForSequenceClassification.from_pretrained(
        f"{MODEL_PATH}{model_type}",
        num_labels=10,
        problem_type="multi_label_classification",
    ).to(device)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Use a conservative max_length value
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,  # Fixed smaller value to avoid overflow
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        batch_probs = torch.sigmoid(outputs.logits).cpu().numpy()
        all_probs.append(batch_probs)

    # Combine all batches
    predicted_probabilities_np = np.vstack(all_probs)

    # We've already processed all batches and stored results in predicted_probabilities_np
    y_pred_np = np.zeros_like(predicted_probabilities_np)

    # Fix the path to use the model parameter instead of model_type
    thresholds_path = f"{MODEL_PATH}{model_type}/best_thresholds.json"
    with open(thresholds_path, "r") as f:
        best_thresholds = json.load(f)

    # Loop over each label index and name in drug_cols
    for idx, label_name in enumerate(drug_cols):
        thr = best_thresholds[label_name]  # get the threshold for this specific label
        y_pred_np[:, idx] = (predicted_probabilities_np[:, idx] >= thr).astype(int)

    # Store predictions in DataFrame
    for idx, label_name in enumerate(drug_cols):
        pred_df[f"predict_prob_{label_name}"] = predicted_probabilities_np[:, idx]
        pred_df[f"pred_{label_name}"] = y_pred_np[:, idx]

    current_time = datetime.now()
    # Format the date and time for a filename
    filename_time = current_time.strftime("%Y%m%d_%H%M")

    pred_df.to_csv(f"{REPORTS_OUTPUT_PATH}{model_type}_outputs_{filename_time}.csv")


@app.command()
def predict(
    input_data: str = typer.Argument(
        ..., help="Path to the CSV file containing the input data"
    ),
    text_col: str = typer.Argument(..., help="Name of the column containing text data"),
    model_type: str = typer.Argument(
        ..., help="Type of BERT model ('BERT' or 'Bio_ClinicalBERT')"
    ),
):
    """
    Generate predictions using a trained BERT model for drug overdose classification.
    """
    start_time = time.time()
    predict_bert(input_data, model_type, text_col)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    app()
