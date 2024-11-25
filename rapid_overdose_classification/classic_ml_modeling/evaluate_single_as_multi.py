from model_tuner import loadObjects
import os
import sys
import pandas as pd


def load_models_for_outcome(embedding_type):

    embedding_type_path = (
        f"../../models/classic_ml_models/single_label/{embedding_type}/"
    )

    # Dictionary to store the models
    models_dict = {}

    for file_name in os.listdir(embedding_type_path):
        if file_name.endswith(".pkl"):
            drug_name = file_name.split("_")[0]

            models_dict[drug_name] = loadObjects(file_name)

    return models_dict


def predict_all_models(text, models_dict):
    """
    Predict for the input text using each model in the dictionary and build a DataFrame
    with prediction column names derived from the model names.

    Args:
        text (str): Input text to predict.
        models_dict (dict): Dictionary containing models with their corresponding drug names as keys.

    Returns:
        pd.DataFrame: DataFrame with predictions for each model.
    """
    # Initialize an empty dictionary to store predictions
    predictions = {}

    for drug_name, model in models_dict.items():
        # Assuming the model has a `predict` method and can take the input text in the appropriate format
        try:
            prediction = model.predict([text])[
                0
            ]  # Predict for the text (wrapping in a list for sklearn models)
            predictions[f"{drug_name}_prediction"] = prediction
        except Exception as e:
            print(f"Error predicting with model {drug_name}: {e}")
            predictions[f"{drug_name}_prediction"] = (
                None  # Handle errors gracefully by adding None
            )

    # Convert predictions dictionary to a DataFrame
    predictions_df = pd.DataFrame([predictions])
    return predictions_df


if __name__ == "__main__":
    import sys

    embedder = sys.argv[1]  # Embedding type as input argument
    text_input = sys.argv[2]  # Text input to predict

    # Load the models
    model_dict = load_models_for_outcome(embedder)

    # Predict using all models
    predictions_df = predict_all_models(text_input, model_dict)
