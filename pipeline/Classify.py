import sys
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


def predict(pred_df):

    device = "cuda"
    # Extract text from input
    texts = pred_df["text"].tolist()
    # Load the correct tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"../../models/bert_models/{model}")

    # Tokenize the text
    inputs = tokenizer(
        texts, return_tensors="pt", padding="max_length", truncation=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        f"../../models/bert_models/{model}",
        num_labels=11,
        problem_type="multi_label_classification",
    ).to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted probabilities
    logits = outputs.logits
    predicted_probabilities = torch.sigmoid(logits)

    # Convert probabilities to binary (0 or 1) predictions
    y_pred = (predicted_probabilities > 0.5).int()

    pred_df["predict_prob"] = predicted_probabilities
    pred_df["pred"] = y_pred

    pred_df.to_csv(f"./pipeline_steps/final_output.csv")


if __name__ == "__main__":

    location_of_file = sys.argv[1]

    input_df = pd.read_csv(location_of_file)

    ## load best model
    predict(input_df)
