import typer
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from rapid_overdose_classification.bert_modeling.constants import (
    drug_cols,
    device,
    MODEL_PATH,
    EVALUATION_RESULTS_PATH,
    EVALUATION_MISMATCHES_PATH,
    EVALUATION_METRICS_PATH,
)
from rapid_overdose_classification.bert_modeling.config import (
    EVAL_BATCH_SIZE,
    MAX_LENGTH,
    BOOTSTRAP_METRICS,
    BOOTSTRAP_N_SAMPLES,
    BOOTSTRAP_NUM_RESAMPLES,
    BOOTSTRAP_AVERAGE,
    EXTERNAL_EXPERIMENT_NAME,
    INTERNAL_EXPERIMENT_NAME,
)
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
from model_tuner.bootstrapper import evaluate_bootstrap_metrics
from tqdm import tqdm
import numpy as np
import mlflow
from rapid_overdose_classification.config import mlflow_uri
from pipeline.regex_classifier import clean_bert_text

app = typer.Typer(help="Evaluation of BERT style models")


def evaluate_bert_models(
    input_data: str,
    model_type: str,
    external_dataset_or_test: int,
    batch_size: int = EVAL_BATCH_SIZE,
):
    """
    Evaluates BERT models on a given dataset with batching.

    Args:
        input_data (str): Path to the CSV or PKL file containing the evaluation data.
        model_type (str): Type of the BERT model to be used for evaluation.
        external_dataset_or_test (int): Flag indicating if this is external dataset (1) or test set (0).
        batch_size (int): Batch size for processing.

    Returns:
        None
    """
    mlflow.set_tracking_uri(mlflow_uri)

    if input_data.endswith(".pkl"):
        eval_df = pd.read_pickle(input_data)
    elif input_data.endswith(".xlsx"):
        eval_df = pd.read_excel(input_data)
    else:
        eval_df = pd.read_csv(input_data)

    texts = eval_df["text"].apply(clean_bert_text).tolist()
    y_true = eval_df[drug_cols].values

    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}{model_type}")
    model = AutoModelForSequenceClassification.from_pretrained(
        f"{MODEL_PATH}{model_type}",
        num_labels=len(drug_cols),
        problem_type="multi_label_classification",
    ).to(device)

    print("Tokenizing inputs...")
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
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
    thresholds_path = f"{MODEL_PATH}{model_type}/best_thresholds.json"
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
    roc_auc = roc_auc_score(y_true_np, predicted_probabilities.numpy(), average="macro")
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
    print(f"Macro AUC ROC: {roc_auc}")

    if external_dataset_or_test:
        experiment_name = EXTERNAL_EXPERIMENT_NAME
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"Finetuned model {model_type}") as parent_run:
            mlflow.log_metric("Hamming Loss", hamming)
            mlflow.log_metric("macro f1", f1)
            mlflow.log_metric("macro roc_auc", roc_auc)
            mlflow.log_metric("accuracy", accuracy)

        results = evaluate_bootstrap_metrics(
            y=y_true_np,
            y_pred_prob=predicted_probabilities_np,
            thresholds=best_thresholds,
            metrics=BOOTSTRAP_METRICS,
            n_samples=BOOTSTRAP_N_SAMPLES,
            num_resamples=BOOTSTRAP_NUM_RESAMPLES,
            average=BOOTSTRAP_AVERAGE,
            balance=False,
        )

        bootstrap_metrics_dict = results.to_dict(orient="records")

        for metric in bootstrap_metrics_dict:
            mlflow.log_metric(f"{metric['Metric']}_mean", metric["Mean"])
            mlflow.log_metric(
                f"{metric['Metric']}_95_CI_low",
                metric["95% CI Lower"],
            )
            mlflow.log_metric(
                f"{metric['Metric']}_95_CI_high",
                metric["95% CI Upper"],
            )

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
            class_roc_auc = roc_auc_score(
                y_true_np[:, i], predicted_probabilities[:, i].numpy()
            )

            # Store metrics
            class_metrics[drug] = {
                "accuracy": class_acc,
                "precision": class_prec,
                "recall": class_rec,
                "f1": class_f1,
                "roc_auc": class_roc_auc,
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

        output_df.to_csv(EVALUATION_RESULTS_PATH)
        mismatches_df.to_csv(EVALUATION_MISMATCHES_PATH)
        metrics_df.to_csv(EVALUATION_METRICS_PATH)

    else:
        experiment_name = INTERNAL_EXPERIMENT_NAME
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"Finetuned model {model_type}") as parent_run:
            mlflow.log_metric("Hamming Loss", hamming)
            mlflow.log_metric("macro f1", f1)
            mlflow.log_metric("accuracy", accuracy)

            results = evaluate_bootstrap_metrics(
                y=y_true_np,
                y_pred_prob=predicted_probabilities_np,
                thresholds=best_thresholds,
                metrics=BOOTSTRAP_METRICS,
                n_samples=BOOTSTRAP_N_SAMPLES,
                num_resamples=BOOTSTRAP_NUM_RESAMPLES,
                average=BOOTSTRAP_AVERAGE,
                balance=False,
            )

            bootstrap_metrics_dict = results.to_dict(orient="records")

            for metric in bootstrap_metrics_dict:
                mlflow.log_metric(f"{metric['Metric']}_mean", metric["Mean"])
                mlflow.log_metric(
                    f"{metric['Metric']}_95_CI_low",
                    metric["95% CI Lower"],
                )
                mlflow.log_metric(
                    f"{metric['Metric']}_95_CI_high",
                    metric["95% CI Upper"],
                )


@app.command()
def evaluate(
    input_data: str = typer.Argument(
        "../../data/test_set.csv", help="Path to input test data (CSV, PKL, or XLSX)"
    ),
    model_type: str = typer.Argument(
        "bioclinicalbert", help="BERT model type ('BERT' or 'Bio_ClinicalBERT')"
    ),
    external_dataset: int = typer.Argument(
        0, help="0 for internal test set, 1 for external dataset"
    ),
    batch_size: int = typer.Argument(EVAL_BATCH_SIZE, help="Batch size for evaluation"),
):
    """
    Evaluate BERT models on test datasets with comprehensive metrics and MLflow logging.
    """
    evaluate_bert_models(input_data, model_type, external_dataset, batch_size)


if __name__ == "__main__":
    app()
