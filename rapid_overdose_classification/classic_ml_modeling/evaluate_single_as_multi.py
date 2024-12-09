from model_tuner import loadObjects
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    hamming_loss,
    roc_auc_score,
)
from sklearn.preprocessing import MultiLabelBinarizer


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


def load_models_for_outcome(embedding_type):

    embedding_type_path = (
        f"../../models/classic_ml_models/single_label/{embedding_type}/"
    )

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


if __name__ == "__main__":
    import sys

    embedder = sys.argv[1]
    text_input = sys.argv[2]

    model_dict = load_models_for_outcome(embedder)

    text_df = pd.read_pickle(text_input)

    if embedder == "bioclinicalbert":
        X = text_df["clinBERTEmbed"]
        X = np.stack(X, axis=0)
        n_samples, sequence_length, n_features = X.shape
        X = X.reshape(-1, n_features)

    elif embedder == "cuis":
        X = text_df["vector"]
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

    # Evaluate metrics
    accuracy = accuracy_score(true_values, predicted_values)
    print(f"Accuracy: {accuracy:.2f}")

    hamming = hamming_loss(true_values, predicted_values)
    print(f"Hamming Loss: {hamming:.2f}")

    print(f"ROC AUC: {roc_auc:.2f}")

    report = classification_report(
        true_values, predicted_values, target_names=y_df.columns
    )
    print("Classification Report:")
    print(report)
