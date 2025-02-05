from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch
import pandas as pd
import numpy as np
import sys
from constants import drug_cols, device
from transformers_interpret import MultiLabelClassificationExplainer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json


def save_explainability(explain_df, model_type):

    ### Loading the finetunedBERT model
    tokenizer = AutoTokenizer.from_pretrained(f"../../models/{model_type}")
    model = AutoModelForSequenceClassification.from_pretrained(
        f"../../models/{model_type}",
        num_labels=len(drug_cols),
        problem_type="multi_label_classification",
    ).to(device)
    texts = explain_df["text"].tolist()
    y_true = explain_df[drug_cols].values

    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    dataset = TensorDataset(
        encodings["input_ids"], encodings["attention_mask"], torch.tensor(y_true)
    )

    dataloader = DataLoader(dataset, batch_size=8)

    ### Defining trainer to use to do the predictions
    ### In the pipeline this will be setup without using the
    ### Trainer class and loading the model only for inference

    print("Predicting on the dataset...")
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        print("Evaluating Batches:")
        for batch in tqdm(dataloader):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    predicted_probabilities = torch.sigmoid(torch.cat(all_logits))
    predicted_probabilities_np = predicted_probabilities.numpy()

    ### Defining the sigmoid curve and using this to
    ### Normalise the predicted outputs
    thresholds_path = f"../../models/{model_type}/best_thresholds.json"
    with open(thresholds_path, "r") as f:
        best_thresholds = json.load(f)

    # create an array for preds
    y_pred_np = np.zeros_like(predicted_probabilities_np)

    # Loop over each label index and name in drug_cols
    for idx, label_name in enumerate(drug_cols):
        thr = best_thresholds[label_name]  # get the threshold for this specific label
        y_pred_np[:, idx] = (predicted_probabilities_np[:, idx] >= thr).astype(int)

    y_true_np = torch.cat(all_labels).numpy()

    multiexplainer = MultiLabelClassificationExplainer(
        model, tokenizer, custom_labels=drug_cols
    )

    explain_df = pd.DataFrame(
        {
            "text": texts,
            "true_labels": [list(row) for row in y_true],  # Convert to list of lists
            "predicted_labels": [list(row) for row in y_pred_np],
        }
    )

    ## Looping through and interpretting each incorrect prediction
    def interpret(text, filename, true_labels):
        word_attributions = multiexplainer(text)
        return multiexplainer.visualize(filename, true_class=list(true_labels))

    word_attributions = multiexplainer(explain_df.loc[0]["text"])
    print("Logging explainability htmls to logs/bioclinicalBERT_explainability/")
    for index, row in explain_df.iterrows():
        interpret(
            explain_df.loc[index]["text"],
            f"../../reports/bioclinicalBERT_explainability/{index}.html",
            true_labels=explain_df.loc[index]["true_labels"],
        )
    print("Done")


if __name__ == "__main__":

    file_to_explain = sys.argv[1]
    model_type = sys.argv[2]

    explain_file = pd.read_csv(file_to_explain)

    save_explainability(explain_file, model_type)
