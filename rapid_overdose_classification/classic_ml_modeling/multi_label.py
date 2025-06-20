import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from model_tuner import Model, train_val_test_split
from model_tuner.pickleObjects import dumpObjects
from rapid_overdose_classification.constants import (
    all_drug_cols,
    COMBINED_DATA_DIR,
    MULTI_LABEL_MODEL_TYPES,
)
from sklearn.metrics import hamming_loss, make_scorer
from sklearnex import patch_sklearn
from sklearn.metrics import multilabel_confusion_matrix
from model_tuner.bootstrapper import evaluate_bootstrap_metrics
import typer
from rapid_overdose_classification.config import (
    MLFLOW_URI,
    BOOTSTRAP_NUM_RESAMPLES,
    BOOTSTRAP_NUM_SAMPLES,
    RANDOM_STATE,
    RANDOM_GRID,
    multi_label_model_parameters,
    MULTI_LABEL_BOOTSTRAP_METRICS,
    MULTI_LABEL_N_ITER,
)


def multi_label_classifier(model_type):
    """
    This function takes model_type as input and will train a multi label classifier
    and log the results to mlflow.

    Args:
        model_type (str): type of multi label classifier to train, either RandomForest
        or XGBoost.
    """
    if model_type not in MULTI_LABEL_MODEL_TYPES:
        raise ValueError(f"Model type must be one of {MULTI_LABEL_MODEL_TYPES}")

    patch_sklearn()
    mlflow.set_tracking_uri(MLFLOW_URI)
    experiment_name = f"Multi Label"
    mlflow.set_experiment(experiment_name)

    drug_df = pd.read_pickle(COMBINED_DATA_DIR)
    hamming = make_scorer(hamming_loss, greater_is_better=False)

    y = drug_df[all_drug_cols].values
    X = drug_df["clinBERTEmbed"].values
    X = np.stack(X, axis=0)
    _, _, n_features = X.shape
    X = X.reshape(-1, n_features)

    with mlflow.start_run(run_name=model_type):

        model = Model(
            name=model_type,
            model_type="classification",
            estimator_name=multi_label_model_parameters[model_type]["estimator_name"],
            multi_label=True,
            class_labels=all_drug_cols,
            custom_scorer={"hamming_loss": hamming},
            calibrate=False,
            estimator=multi_label_model_parameters[model_type]["estimator"],
            kfold=False,
            stratify_y=False,
            grid=multi_label_model_parameters[model_type]["tuned_parameters"],
            randomized_grid=RANDOM_GRID,
            boost_early=multi_label_model_parameters[model_type]["xgbearly"],
            n_iter=MULTI_LABEL_N_ITER,
            scoring=["hamming_loss"],
            n_jobs=-2,
            random_state=RANDOM_STATE,
        )

        print(f"Tuning hyperparameters for all drugs:")
        model.grid_search_param_tuning(X, y)

        X_train, X_valid, X_test, y_train, y_valid, y_test = train_val_test_split(
            X,
            y,
            stratify_y=False,
            random_state=RANDOM_STATE,
            train_size=model.train_size,
            validation_size=model.validation_size,
            test_size=model.test_size,
        )

        model.kfold = False

        y_prob = model.predict_proba(X_test)
        # Reshaping and extracting just the predicted positive class
        # for y_prob
        y_prob = np.array([label_probs[:, 1] for label_probs in y_prob]).T
        ### F1 Weighted
        y_pred = model.predict(X_test, optimal_threshold=False)
        f1 = f1_score(y_test, y_pred, average="macro")
        ### Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        ### validation metrics test
        y_pred_valid = model.predict(X_valid, optimal_threshold=False)
        conf_valid = multilabel_confusion_matrix(y_valid, y_pred_valid)

        hamming_l = hamming_loss(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob, average="macro")

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
        mlflow.log_metric("Macro F1 Test", f1)
        mlflow.log_metric("Accuracy Test", accuracy)
        mlflow.log_metric("ROC AUC", roc_auc)

        best_thresholds = [model.threshold["hamming_loss"]] * len(all_drug_cols)

        results = evaluate_bootstrap_metrics(
            y=y_test,
            y_pred_prob=y_prob,
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

        dumpObjects(
            model, f"../../models/classic_ml_models/multi_label/{model_type}.pkl"
        )


app = typer.Typer(help="Training of multi-label classifiers")


@app.command()
def main(
    model_type: str = typer.Argument(
        "RandomForest",
        help="Type of multi label classifier to train (RandomForest or XGBoost)",
    ),
):
    multi_label_classifier(model_type)


if __name__ == "__main__":
    app()
