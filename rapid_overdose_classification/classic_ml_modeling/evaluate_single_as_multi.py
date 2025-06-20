from model_tuner import loadObjects
from model_tuner.bootstrapper import evaluate_bootstrap_metrics
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    hamming_loss,
    roc_auc_score,
    f1_score,
)
import mlflow
import typer
from rapid_overdose_classification.constants import (
    drug_cols,
    TEST_SET_PATH,
    EMBEDDING_TYPES,
    MODEL_PATH_TEMPLATE,
)
from rapid_overdose_classification.config import (
    MLFLOW_URI,
    BOOTSTRAP_NUM_RESAMPLES,
    BOOTSTRAP_NUM_SAMPLES,
    MULTI_LABEL_BOOTSTRAP_METRICS,
    MODEL_SCORING_METRIC,
)


def load_models_for_outcome(embedding_type):

    embedding_type_path = MODEL_PATH_TEMPLATE.format(embedding_type=embedding_type)

    # Dictionary to store the models
    models_dict = {}

    for file_name in os.listdir(embedding_type_path):
        if file_name.endswith(".pkl"):
            drug_name = file_name.split("_")[0]

            model_loc = embedding_type_path + file_name

            models_dict[drug_name] = loadObjects(model_loc)

    return models_dict


def predict_all_models(X_column, models_dict):
    """
    Predict for the input text using each model in the dictionary and build a DataFrame
    with prediction column names derived from the model names.

    Args:
        text (str): Input text to predict.
        models_dict (dict): Dictionary containing models with their corresponding drug names as keys.

    Returns:
        pd.DataFrame: DataFrame with predictions for each model.
    """
    predictions = {}
    probabilities = {}
    for drug_name, model in models_dict.items():
        try:
            # print(X_column)
            prediction = model.predict(X_column)
            proba = model.predict_proba(X_column)[:, 1]
            predictions[f"{drug_name}"] = prediction
            probabilities[f"{drug_name}"] = proba
        except Exception as e:
            print(f"Error predicting with model {drug_name}: {e}")
            predictions[f"{drug_name}"] = (
                None  # Handle errors gracefully by adding None
            )

    # Convert predictions dictionary to a DataFrame
    predictions_df = pd.DataFrame([predictions])
    probabilities_df = pd.DataFrame([probabilities])
    # Transpose the dataframe
    transposed_data = {
        col: pd.Series(predictions_df[col][0]) for col in predictions_df.columns
    }
    predictions_df = pd.DataFrame(transposed_data)

    # Transpose the dataframe
    transposed_data_prob = {
        col: pd.Series(probabilities_df[col][0]) for col in probabilities_df.columns
    }
    probabilities_df = pd.DataFrame(transposed_data_prob)

    print(predictions_df.columns)

    return predictions_df, probabilities_df


def evaluate_classic_models(embedder: str, text_input: str):
    mlflow.set_tracking_uri(MLFLOW_URI)

    model_dict = load_models_for_outcome(embedder)

    text_df = pd.read_pickle(text_input)

    if embedder == "bioclinicalbert":
        X = text_df["clinBERTEmbed"]
        X = np.stack(X, axis=0)
        n_samples, sequence_length, n_features = X.shape
        X = X.reshape(-1, n_features)
    elif embedder == "cuis":
        X = text_df["vector"].values
        default_array = np.zeros(len(X[2]))
        cleaned_X = [
            np.array(entry) if isinstance(entry, list) else default_array for entry in X
        ]
        X = np.stack(cleaned_X, axis=0)
    elif embedder == "glove":
        X = text_df["GloVE_proc"]

    predictions_df, probabilities_df = predict_all_models(X, model_dict)

    y_df = text_df[drug_cols]

    predictions_df = predictions_df[y_df.columns]
    probabilities_df = probabilities_df[y_df.columns]

    predicted_values = predictions_df.to_numpy()
    true_values = y_df.to_numpy()
    probability_values = probabilities_df.to_numpy()

    # Optional: Check shapes and data types
    print("Predicted Values Shape:", predicted_values.shape)
    print("True Values Shape:", true_values.shape)
    print("Predicted Values Data Type:", predicted_values.dtype)
    print("True Values Data Type:", true_values.dtype)

    roc_auc = roc_auc_score(true_values, probability_values, average="macro")
    f1 = f1_score(true_values, predicted_values, average="macro")

    # Evaluate metrics
    accuracy = accuracy_score(true_values, predicted_values)
    print(f"Accuracy: {accuracy:.3f}")

    hamming = hamming_loss(true_values, predicted_values)
    print(f"Hamming Loss: {hamming:.3f}")

    print(f"Macro ROC AUC: {roc_auc:.3f}")
    print(f"Macro F1 Score: {f1:.3f}")

    report = classification_report(
        true_values, predicted_values, target_names=y_df.columns
    )
    print("Classification Report:")
    print(report)

    experiment_name = f"Table 3 Results"
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"Single Label as Multi {embedder}") as parent_run:
        mlflow.log_metric("Hamming Loss", hamming)
        mlflow.log_metric("macro f1", f1)
        mlflow.log_metric("macro roc_auc", roc_auc)
        mlflow.log_metric("accuracy", accuracy)

        best_thresholds = [
            model_dict[col].threshold[MODEL_SCORING_METRIC]
            for col in probabilities_df.columns
        ]

        results = evaluate_bootstrap_metrics(
            y=true_values,
            y_pred_prob=probability_values,
            thresholds=best_thresholds,
            metrics=MULTI_LABEL_BOOTSTRAP_METRICS,
            n_samples=BOOTSTRAP_NUM_SAMPLES,
            num_resamples=BOOTSTRAP_NUM_RESAMPLES,
            average="macro",
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


app = typer.Typer(
    help="Evaluation of classic ML models (evaluating single as multi label)"
)


@app.command()
def main(
    embedder: str = typer.Argument(
        "bioclinicalbert", help=f"Embedding type ({', '.join(EMBEDDING_TYPES)})"
    ),
    text_input: str = typer.Argument(
        TEST_SET_PATH, help="Path to the input text data file"
    ),
):
    if embedder not in EMBEDDING_TYPES:
        raise ValueError(f"Embedder must be one of {EMBEDDING_TYPES}")
    evaluate_classic_models(embedder, text_input)


if __name__ == "__main__":
    app()
