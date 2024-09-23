import sys
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

drug_cols = [
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
    "Drug No Opioids",
]


def predict(pred_df, model):

    device = "cuda"
    # Extract text from input
    texts = pred_df["text"].tolist()
    # Load the correct tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"../models/{model}/")

    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    model = AutoModelForSequenceClassification.from_pretrained(
        f"../models/{model}",
        num_labels=11,
        problem_type="multi_label_classification",
    ).to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted probabilities
    logits = outputs.logits
    predicted_probabilities = torch.sigmoid(logits).cpu()

    # Convert probabilities to binary (0 or 1) predictions
    y_pred = (predicted_probabilities > 0.5).int()

    predicted_df = pd.DataFrame(y_pred, columns=drug_cols)

    output_df = pd.concat([pred_df, predicted_df], axis=1)

    output_df.to_csv(f"./pipeline_steps/final_output.csv")


def create_text_col(input_df):

    cause_alphabet = ["CauseA", "CauseB", "CauseC"]
    cause_secondary = ["Primary Cause", "Secondary Cause"]

    if any(x in cause_alphabet for x in input_df.columns):
        input_df["text"] = (
            input_df["CauseA"].astype(str) + ", " + input_df["CauseB"].astype(str)
        )

    if any(x in cause_secondary for x in input_df.columns):
        input_df["text"] = (
            input_df["Primary Cause"].astype(str)
            + ", "
            + input_df["Secondary Cause"].astype(str)
        )

    return input_df


if __name__ == "__main__":

    location_of_file = sys.argv[1]

    model = sys.argv[2]

    input_df = pd.read_csv(location_of_file)

    input_df = create_text_col(input_df)

    ## load best model
    predict(input_df, model)
