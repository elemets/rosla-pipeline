import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay
import sys
from model_tuner import Model
from model_tuner.pickleObjects import dumpObjects
from constants import all_drug_cols
from sklearn.metrics import hamming_loss, make_scorer
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearnex import patch_sklearn
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def multi_label_classifier(model_type):
    """
    This function takes model_type as input and will train a multi label classifier
    and log the results to mlflow.

    Args:
        model_type (str): type of multi label classifier to train, either RandomForest
        or XGBoost.
    """

    patch_sklearn()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment_name = f"Multi Label"
    mlflow.set_experiment(experiment_name)

    if model_type == "XGBoost":
        estimator = XGBClassifier()

        estimator_name = "xgb"
        ### need to do this because we have to nest it in  a mutlioutputclassifier
        tuned_parameters = {
            f"{estimator_name}__estimator__max_depth": [3, 5, 10],
            f"{estimator_name}__estimator__learning_rate": [0.03, 0.003],
            f"{estimator_name}__estimator__n_estimators": [50, 10, 100],
            f"{estimator_name}__estimator__n_jobs": [-2],
            f"{estimator_name}__estimator__device": ["cuda"],
            f"{estimator_name}__estimator__early_stopping_rounds": [10],
            f"{estimator_name}__estimator__eval_metric": ["logloss"],
        }
        estimator = MultiOutputClassifier(estimator)
    elif model_type == "RandomForest":
        rf = RandomForestClassifier(class_weight="balanced")

        estimator_name = "rf"

        tuned_parameters = {
            f"{estimator_name}__estimator__max_depth": [3, 5, 10, None],
            f"{estimator_name}__estimator__n_estimators": [10, 100, 200],
            f"{estimator_name}__estimator__max_features": [1, 3, 5, 7],
            f"{estimator_name}__estimator__min_samples_leaf": [1, 2, 3],
        }

        estimator = MultiOutputClassifier(rf)
    else:
        raise ("Need to specify a model type out of RandomForest and XGBoost")

    drug_df = pd.read_pickle("../../data/outcomes_squashed/outcomes_squashed.pkl")
    hamming = make_scorer(hamming_loss, greater_is_better=False)

    y = drug_df[all_drug_cols].values
    X = drug_df["clinBERTEmbed"].values
    X = np.stack(X, axis=0)
    _, _, n_features = X.shape
    X = X.reshape(-1, n_features)
    with mlflow.start_run(run_name=model_type):

        model = Model(
            name=model_type,
            estimator_name=estimator_name,
            multi_label=True,
            class_labels=all_drug_cols,
            custom_scorer={"hamming_loss": hamming},
            calibrate=False,
            estimator=estimator,
            kfold=False,
            stratify_y=False,
            grid=tuned_parameters,
            randomized_grid=False,
            n_iter=3,
            scoring=["hamming_loss"],
            n_jobs=-2,
            random_state=42,
        )

        print(f"Tuning hyperparameters for all drugs:")

        model.grid_search_param_tuning(X, y)

        X_train, X_valid, X_test, y_train, y_valid, y_test = model.train_val_test_split(
            X,
            y,
            stratify_y=False,
            random_state=42,
            train_size=model.train_size,
            validation_size=model.validation_size,
            test_size=model.test_size,
            calibrate=False,
            stratify_cols=model.stratify_cols,
        )

        model.kfold = False

        y_prob = model.predict_proba(X_test)

        ### F1 Weighted
        y_pred = model.predict(X_test, optimal_threshold=False)
        f1 = f1_score(y_test, y_pred, average="weighted")
        ### Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        ### validation metrics test
        y_pred_valid = model.predict(X_valid, optimal_threshold=False)
        conf_valid = multilabel_confusion_matrix(y_valid, y_pred_valid)

        hamming_l = hamming_loss(y_test, y_pred)
        for index, cm in enumerate(conf_valid):
            print(cm)
            cm_valid = ConfusionMatrixDisplay(cm)

            fig, ax = plt.subplots(figsize=(10, 10))
            cm_valid.plot(
                ax=ax, values_format="d"
            )  # values_format is optional, for integer display
            plt.title("Confusion Matrix")

            plt.title(f"Validation CM for {all_drug_cols[index]}")
            plt.show()
            plt.close()
            mlflow.log_figure(
                fig,
                f"confusion matrix {model_type} {all_drug_cols[index]}.png",
            )

        plt.title(f"Confusion Matrix for: {model_type} on test set")
        plt.show()

        for param, value in model.best_params_per_score[model.scoring[0]][
            "params"
        ].items():
            mlflow.log_param(param, value)

        mlflow.log_metric("Hamming Test", hamming_l)
        mlflow.log_metric("F1 Test", f1)
        mlflow.log_metric("Accuracy Test", accuracy)
        class_report = model.classification_report
        for class_or_avg, metrics_dict in class_report.items():
            for metric, value in metrics_dict.items():
                mlflow.log_metric(class_or_avg + "_validation_" + metric, value)

        dumpObjects(
            model, f"../../models/models/classic_ml_models/multi_label/{model_type}.pkl"
        )


if __name__ == "__main__":
    model_type = sys.argv[1]
    multi_label_classifier(model_type)
