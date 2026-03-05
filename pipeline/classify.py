import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse

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
]


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        # Convert lists to tensors
        encoding = {key: torch.tensor(val) for key, val in encoding.items()}
        return encoding


def predict(pred_df, model_name, batch_size=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count()

    # Extract text from input
    texts = pred_df["text"].tolist()

    # Load the correct tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"../models/{model_name}/")

    # Create Dataset and DataLoader for batch processing
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Load the model and wrap it for multi-GPU
    model = AutoModelForSequenceClassification.from_pretrained(
        f"../models/{model_name}",
        num_labels=10,
        problem_type="multi_label_classification",
    )

    if n_gpus > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)

    all_predictions = []

    # Batch-wise prediction with multiple GPUs
    with torch.no_grad():
        print("Number of batches:")
        for batch in tqdm(dataloader):
            # Send tensors directly to device
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_probabilities = torch.sigmoid(logits).cpu()
            all_predictions.append(predicted_probabilities)

    # Concatenate all batch results
    y_pred = torch.cat(all_predictions, dim=0)
    y_pred = (y_pred > 0.5).int()

    # Convert to DataFrame
    predicted_df = pd.DataFrame(y_pred.numpy(), columns=drug_cols)

    # Ensure indices are aligned correctly
    output_df = pd.concat(
        [pred_df.reset_index(drop=True), predicted_df.reset_index(drop=True)], axis=1
    )

    return output_df


def create_text_col(input_df):
    # Remove spaces from columns for checking
    standardized_columns = input_df.columns.str.replace(" ", "")
    input_df.columns = standardized_columns  # Update columns to be without spaces

    cause_alphabet = ["CauseA", "CauseB", "CauseC"]
    cause_secondary = ["PrimaryCause", "SecondaryCause"]


    if any(x in standardized_columns for x in cause_alphabet):
        input_df["text"] = (
            input_df["CauseA"].astype(str) + ", " + input_df["CauseB"].astype(str) + ", "
        )

    elif any(x in standardized_columns for x in cause_secondary):
        input_df["text"] = (
            input_df["PrimaryCause"].astype(str)
            + ", "
            + input_df["SecondaryCause"].astype(str)
        )

    elif ['InjuryDesc'] in standardized_columns:
        input_df['text'] = input_df['text'].astype(str) + input_df['InjuryDesc'].astype(str)


    # Cleaning text column making sure we deal with typos

    else:
        raise ValueError("Required cause columns are missing in the input DataFrame.")

    return input_df

def classify_file(df: str, output_path: str, model_name: str, batch_size: int = 1024):

    df = create_text_col(df)
    pred_df = predict(df, model_name, batch_size=batch_size)

    pred_df.to_csv(output_path, index=False)

    return pred_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify text data.")
    parser.add_argument("-i", "--input", required=True, help="Input CSV/XLSX file.")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file.")
    parser.add_argument("-m", "--model", required=True, help="Model name or path.")

    args = parser.parse_args()

    location_of_file = args.input
    output_name = args.output
    model_name = args.model

    if location_of_file.endswith(".csv"):
        input_df = pd.read_csv(location_of_file)
    elif location_of_file.endswith(".xlsx"):
        input_df = pd.read_excel(location_of_file)
    else:
        raise ValueError("Please specify either a .csv or .xlsx file")

    input_df = create_text_col(input_df)

    # Predict on the dataset with batch size to handle large input
    pred_df = predict(input_df, model_name, batch_size=1024)
    output_df = pred_df[pred_df["Any Drugs"] != 0].reset_index(drop=True)
    output_df["source_file"] = str(location_of_file)
    # Saving the results to CSV
    output_df.to_csv(f"{output_name}")
