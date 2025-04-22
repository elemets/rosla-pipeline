import pandas as pd
import sys
from datasets import Dataset
from constants import device, drug_cols
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import numpy as np
import torch
from datetime import datetime
import json


def predict_bert(input_data, model, text_col):
    """
    Predicts labels for a given dataset using a BERT model.

    Args:
        input_data (str): Path to the CSV file containing the input data.
        model (str): Type of the BERT model to be used for prediction.
        text_col (str): Name of the column containing text data in the input CSV file.

    Returns:
        None

    This function performs the following steps:
    1. Loads the input data from a CSV file.
    2. Extracts text data from the specified column.
    3. Loads the appropriate BERT tokenizer.
    4. Tokenizes the text data.
    5. Loads the pre-trained BERT model for sequence classification.
    6. Makes predictions on the input data.
    7. Converts predicted probabilities to binary predictions.
    8. Adds the predicted probabilities and binary predictions to the DataFrame.
    9. Saves the predictions to a CSV file with a timestamped filename.
    """
    # Load data to do the prediction on
    pred_df = pd.read_csv(input_data)
    texts = pred_df[text_col].tolist()
    tokenizer = AutoTokenizer.from_pretrained(f"../../models/bert_models/{model}")

    inputs = tokenizer(
        texts, return_tensors="pt", padding="max_length", truncation=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        f"../../models/bert_models/{model}",
        num_labels=10,
        problem_type="multi_label_classification",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_probabilities = torch.sigmoid(logits)
    predicted_probabilities_np = predicted_probabilities.numpy()
    y_pred_np = np.zeros_like(predicted_probabilities_np)
    thresholds_path = f"../../models/bert_models/{model_type}/best_thresholds.json"
    with open(thresholds_path, "r") as f:
        best_thresholds = json.load(f)

    # Loop over each label index and name in drug_cols
    for idx, label_name in enumerate(drug_cols):
        thr = best_thresholds[label_name]  # get the threshold for this specific label
        y_pred_np[:, idx] = (predicted_probabilities_np[:, idx] >= thr).astype(int)

    pred_df["predict_prob"] = predicted_probabilities
    pred_df["pred"] = y_pred_np
    current_time = datetime.now()

    # Format the date and time for a filename
    filename_time = current_time.strftime("%Y%m%d_%H%M")

    pred_df.to_csv(
        f"../../reports/model_outputs/{model_type}_outputs_{filename_time}.csv"
    )


if __name__ == "__main__":
    input_data_loc = sys.argv[1]
    text_col = sys.argv[2]
    model_type = sys.argv[3]
    predict_bert(input_data_loc, model_type, text_col)
