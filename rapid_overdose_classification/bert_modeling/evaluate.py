import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from helpers import drug_cols
from constants import device
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    hamming_loss,
)
import json

from tqdm import tqdm
import numpy as np
import mlflow


def evaluate_bert_models(
    input_data, model_type, external_dataset_or_test, batch_size=16
):
    """
    Evaluates BERT models on a given dataset with batching.

    Args:
        input_data (str): Path to the CSV or PKL file containing the evaluation data.
        model_type (str): Type of the BERT model to be used for evaluation.
        batch_size (int): Batch size for processing.

    Returns:
        None
    """

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    if input_data.endswith(".pkl"):
        eval_df = pd.read_pickle(input_data)
    elif input_data.endswith(".xlsx"):
        eval_df = pd.read_excel(input_data)
    else:
        eval_df = pd.read_csv(input_data)

    texts = eval_df["text"].tolist()
    y_true = eval_df[drug_cols].values

    tokenizer = AutoTokenizer.from_pretrained(f"../../models/bert_models/{model_type}")
    model = AutoModelForSequenceClassification.from_pretrained(
        f"../../models/bert_models/{model_type}",
        num_labels=len(drug_cols),
        problem_type="multi_label_classification",
    ).to(device)

    print("Tokenizing inputs...")
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

    dataloader = DataLoader(dataset, batch_size=batch_size)

    print("Evaluating the dataset...")
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

    # Create an array for predictions
    y_pred_np = np.zeros_like(predicted_probabilities_np)
    thresholds_path = f"../../models/bert_models/{model_type}/best_thresholds.json"
    with open(thresholds_path, "r") as f:
        best_thresholds = json.load(f)

    # Loop over each label index and name in drug_cols
    for idx, label_name in enumerate(drug_cols):
        thr = best_thresholds[label_name]  # get the threshold for this specific label
        y_pred_np[:, idx] = (predicted_probabilities_np[:, idx] >= thr).astype(int)

    y_true_np = torch.cat(all_labels).numpy()
    print(predicted_probabilities)
    print(y_true_np)
    print(y_pred_np)
    # roc_auc = roc_auc_score(y_true_np, predicted_probabilities.numpy(), average="macro")
    accuracy = accuracy_score(y_true_np, y_pred_np)
    hamming = hamming_loss(y_true_np, y_pred_np)
    precision = precision_score(y_true_np, y_pred_np, average="macro")
    recall = recall_score(y_true_np, y_pred_np, average="macro")
    f1 = f1_score(y_true_np, y_pred_np, average="macro")

    # Print evaluation results
    print(f"Hamming Loss: {hamming}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Macro F1 Score: {f1}")
    # print(f"Macro AUC ROC: {roc_auc}")

    if int(external_dataset_or_test):
        experiment_name = f"External Dataset"
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"Finetuned model {model_type}") as parent_run:
            mlflow.log_metric("Hamming Loss", hamming)
            mlflow.log_metric("macro f1", f1)
            # mlflow.log_metric("macro roc_auc", roc_auc)
            mlflow.log_metric("accuracy", accuracy)

        output_df = pd.DataFrame({"text": texts})

        class_metrics = {}

        for i, drug in enumerate(drug_cols):
            output_df[drug] = y_true_np[:, i]
            output_df[f"{drug}_pred"] = y_pred_np[:, i]
            output_df[f"{drug}_prob"] = predicted_probabilities[:, i]

            # Calculate metrics for this class
            class_acc = accuracy_score(y_true_np[:, i], y_pred_np[:, i])
            class_prec = precision_score(y_true_np[:, i], y_pred_np[:, i])
            class_rec = recall_score(y_true_np[:, i], y_pred_np[:, i])
            class_f1 = f1_score(y_true_np[:, i], y_pred_np[:, i])
            # class_roc_auc = roc_auc_score(
            #     y_true_np[:, i], predicted_probabilities[:, i].numpy()
            # )

            # Store metrics
            class_metrics[drug] = {
                "accuracy": class_acc,
                "precision": class_prec,
                "recall": class_rec,
                "f1": class_f1,
                # "roc_auc": class_roc_auc,
            }

        for i, drug in enumerate(drug_cols):
            output_df[drug] = y_true_np[:, i]
            output_df[f"{drug}_pred"] = y_pred_np[:, i]
            output_df[f"{drug}_prob"] = predicted_probabilities[:, i]

        mismatch_mask = False
        for drug in drug_cols:
            mismatch_mask = mismatch_mask | (
                output_df[drug] != output_df[f"{drug}_pred"]
            )

        # Create a metrics summary DataFrame
        metrics_df = pd.DataFrame(class_metrics).transpose()
        print("\nMetrics Summary:")
        print(metrics_df.round(4))

        mismatches_df = output_df[mismatch_mask].copy()

        mismatches_df = mismatches_df.sort_values("text")

        output_df.to_csv("../../reports/evaluated_res_internal_test_removedmislabels_n_model.csv")

        mismatches_df.to_csv("../../reports/predicted_wrong_internal_test_removedmislabels_n_model.csv")

        metrics_df.to_csv("../../reports/eval_metric_internal_test_removedmislabels_n_model.csv")

    else:
        experiment_name = f"Table 3 Results"
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"Finetuned model {model_type}") as parent_run:
            mlflow.log_metric("Hamming Loss", hamming)
            mlflow.log_metric("macro f1", f1)
            # mlflow.log_metric("macro roc_auc", roc_auc)
            mlflow.log_metric("accuracy", accuracy)


if __name__ == "__main__":
    input_data = sys.argv[1]
    model_type = sys.argv[2]
    external_dataset = sys.argv[3]
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    evaluate_bert_models(input_data, model_type, external_dataset, batch_size)
