import sys
import pandas as pd
import torch
from helpers import (
    drug_cols,
)
from constants import device
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def evaluate_bert_models(input_data, model_type):
    """
    Evaluates BERT models on a given dataset.

    Args:
        input_data (str): Path to the CSV file containing the evaluation data.
        model_type (str): Type of the BERT model to be used for evaluation.

    Returns:
        None

    This function performs the following steps:
    1. Loads the evaluation data from a CSV file.
    2. Tokenizes the text data using the specified BERT tokenizer.
    3. Loads the pre-trained BERT model for sequence classification.
    4. Makes predictions on the evaluation data.
    5. Converts predicted probabilities to binary predictions.
    6. Calculates evaluation metrics (accuracy, precision, recall, F1 score).
    7. Prints the evaluation results.
    """
    # Load data to do the prediction on
    eval_df = pd.read_csv(input_data)

    texts = eval_df["text"].to_list()
    y_true = eval_df[drug_cols].to_list()

    tokenizer = AutoTokenizer.from_pretrained(f"../../models/bert_models/{model_type}")

    inputs = tokenizer(
        texts, return_tensors="pt", padding="max_length", truncation=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        f"../../models/bert_models/{model_type}",
        num_labels=11,
        problem_type="multi_label_classification",
    ).to(device)

    # Make predictions
    print("Evaluating the dataset")
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted probabilities
    logits = outputs.logits
    predicted_probabilities = torch.sigmoid(logits)

    # Convert probabilities to binary (0 or 1) predictions
    y_pred = (predicted_probabilities > 0.5).int()

    # Convert predictions and true labels to numpy for sklearn
    y_pred_np = y_pred.numpy()
    y_true_np = y_true.astype(int)  # Ensure true labels are integers

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true_np, y_pred_np)
    precision = precision_score(
        y_true_np, y_pred_np, average="micro"
    )  # 'micro' for multi-label
    recall = recall_score(y_true_np, y_pred_np, average="micro")
    f1 = f1_score(y_true_np, y_pred_np, average="micro")

    # Print evaluation results
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


if __name__ == "__main__":
    input_data = sys.argv[1]
    model_type = sys.argv[2]
    evaluate_bert_models(input_data, model_type)
