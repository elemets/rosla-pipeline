import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse

try:
    from .regex_classifier import DEFAULT_NFLIS, apply_corrections, build_patterns, clean_bert_text
except ImportError:
    from regex_classifier import DEFAULT_NFLIS, apply_corrections, build_patterns, clean_bert_text

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
        text = clean_bert_text(self.texts[idx])
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
    # Standardize column names by removing spaces, so "Cause A" and "CauseA"
    # (and "Other Cause" / "OtherCause", etc.) collapse to the same name.
    input_df.columns = input_df.columns.str.replace(" ", "")

    # All possible cause columns we want to include (post space-removal).
    # Order here is the order they'll appear in the concatenated text.
    cause_columns = [
        "CauseA",
        "CauseB",
        "CauseC",
        "CauseD",
        "CauseOther",
        "OtherCause",
        "HowInjuryOccurred",
        "InjuryDesc",
    ]

    # Keep only the columns actually present, preserving the order above.
    present_cols = [c for c in cause_columns if c in input_df.columns]

    if not present_cols:
        raise ValueError(
            f"None of the expected cause columns are present in the input. "
            f"Expected at least one of: {cause_columns}"
        )

    # Build the text column by joining the present cause values per row,
    # skipping NaN / empty pieces so we don't end up with ", , something".
    subset = (
        input_df[present_cols]
        .fillna("")
        .astype(str)
        .apply(lambda s: s.str.strip())
    )
    input_df["text"] = subset.apply(
        lambda row: ", ".join(part for part in row if part),
        axis=1,
    ).apply(clean_bert_text)

    return input_df


def apply_regex_classifier(
    pred_df: pd.DataFrame, nflis_path=DEFAULT_NFLIS, verbose: bool = True
) -> pd.DataFrame:
    patterns = build_patterns(nflis_path)
    return apply_corrections(pred_df, patterns, verbose=verbose)


def classify_file(
    df: str,
    output_path: str,
    model_name: str,
    batch_size: int = 1024,
    use_regex: bool = True,
    nflis_path=DEFAULT_NFLIS,
):

    df = create_text_col(df)
    pred_df = predict(df, model_name, batch_size=batch_size)
    if use_regex:
        pred_df = apply_regex_classifier(pred_df, nflis_path=nflis_path)

    pred_df.to_csv(output_path, index=False)

    return pred_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify text data.")
    parser.add_argument("-i", "--input", required=True, help="Input CSV/XLSX file.")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file.")
    parser.add_argument("-m", "--model", required=True, help="Model name or path.")
    parser.add_argument(
        "--skip-regex",
        action="store_true",
        help="Skip the NFLIS-backed regex supplement/correction pass.",
    )
    parser.add_argument(
        "--nflis",
        default=str(DEFAULT_NFLIS),
        help="NFLIS substances CSV used by the regex classifier.",
    )

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
    if not args.skip_regex:
        pred_df = apply_regex_classifier(pred_df, nflis_path=args.nflis)

    output_df = pred_df[pred_df["Any Drugs"] != 0].reset_index(drop=True)
    output_df["source_file"] = str(location_of_file)

    # Saving the results to CSV
    output_df.to_csv(f"{output_name}", index=False)
