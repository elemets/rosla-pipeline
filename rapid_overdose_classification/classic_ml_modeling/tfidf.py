from rapid_overdose_classification.constants import (
    model_list,
    PROCESSED_DATA_TFIDF_DIR,
)
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from model_tuner import Model, dumpObjects
import sklearnex
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from NaiveSVC import NaivelyCalibratedLinearSVC
from rapid_overdose_classification.config import (
    MLFLOW_URI,
    BOOTSTRAP_METRICS,
    model_parameters,
    N_ITER,
    N_SPLITS,
    RANDOM_GRID,
    RANDOM_STATE,
    STRATIFY_Y,
    BOOTSTRAP_NUM_RESAMPLES,
    BOOTSTRAP_NUM_SAMPLES,
    OPTIMAL_THRESH,
    MODEL_SCORING_METRIC,
    KFOLD,
)
import typer


"""
Loops explained:
These models are selected from the model_list and a grid search is performed for
each of these models.
This format is chosen as it gives the most readable results in MLFlow.
We can directly compare XGBoost vs Random Forest for Methamphetamine.

This loop is repeated 4 times for each of the different embedding steps. 
These could also be split into separate .py files, but to run this overnight etc it's 
easier to have one file. 
"""


def tf_idf_single_label(drug):
    sklearnex.patch_sklearn()
    mlflow.set_tracking_uri(MLFLOW_URI)

    experiment_name = f"TFIDF Bootstrapped"
    mlflow.set_experiment(experiment_name)
    drug_df = pd.read_pickle(PROCESSED_DATA_TFIDF_DIR)

    with mlflow.start_run(run_name=f"{drug}") as parent_run:

        best_average_precision = 0
        best_model = 0

        for model_name in tqdm(model_list):
            with mlflow.start_run(run_name=f"{model_name}", nested=True) as child_run:

                y = drug_df[drug].values
                X = drug_df["string_text"].values
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=RANDOM_STATE
                )

                if model_name == "XGBoost":
                    positive_count = np.sum(y)
                    negative_count = len(y) - positive_count
                    scale_pos_weight = negative_count / positive_count

                    estimator = XGBClassifier(
                        objective="binary:logistic", scale_pos_weight=scale_pos_weight
                    )

                else:
                    estimator = clone(model_parameters[model_name]["estimator"])

                model = Model(
                    name=f"{model_name}",
                    estimator_name=model_parameters[model_name]["estimator_name"],
                    calibrate=False,
                    estimator=clone(estimator),
                    kfold=KFOLD,
                    stratify_y=STRATIFY_Y,
                    grid=model_parameters[model_name]["tuned_parameters"],
                    randomized_grid=RANDOM_GRID,
                    n_iter=N_ITER,
                    scoring=[MODEL_SCORING_METRIC],
                    boost_early=model_parameters[model_name]["xgbearly"],
                    pipeline_steps=[("tf_idf", TfidfVectorizer())],
                    n_splits=N_SPLITS,
                    n_jobs=-2,
                    random_state=RANDOM_STATE,
                )

                print(f"Tuning hyperparameters for: {drug}")

                model.grid_search_param_tuning(
                    X_train, y_train, f1_beta_tune=OPTIMAL_THRESH
                )
                model.fit(X_train, y_train, score=MODEL_SCORING_METRIC)
                model.return_metrics(X_train, y_train, optimal_threshold=OPTIMAL_THRESH)

                classreport = model.classification_report

                mlflow.log_metric(
                    "f1_score_valid", classreport["weighted avg"]["f1-score"]
                )
                mlflow.log_metric(
                    "precision_valid", classreport["weighted avg"]["precision"]
                )
                mlflow.log_metric("recall_valid", classreport["weighted avg"]["recall"])

                model.kfold = False

                y_prob = model.predict_proba(X_test)[:, 1]

                bootstrap_metrics = model.return_bootstrap_metrics(
                    X_test,
                    y_test,
                    BOOTSTRAP_METRICS,
                    num_resamples=BOOTSTRAP_NUM_RESAMPLES,
                    n_samples=BOOTSTRAP_NUM_SAMPLES,
                    threshold=model.threshold[MODEL_SCORING_METRIC],
                    balance=False,
                )
                bootstrap_metrics_dict = bootstrap_metrics.to_dict(orient="records")

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

                y_pred = model.predict(X_test, optimal_threshold=OPTIMAL_THRESH)

                cm = confusion_matrix(y_test, y_pred)
                cm_display = ConfusionMatrixDisplay(cm)
                cm_display.plot()
                plt.title(f"Confusion Matrix for: {drug} on test set")
                mlflow.log_figure(cm_display.figure_, f"confusion matrix {drug}.png")
                for param, value in model.best_params_per_score[model.scoring[0]][
                    "params"
                ].items():
                    mlflow.log_param(param, value)

                if classreport["weighted avg"]["f1-score"] > best_average_precision:
                    best_average_precision = classreport["weighted avg"]["f1-score"]
                    best_model = model
                    best_model_type = model_name

            dumpObjects(
                best_model,
                f"../../models/classic_ml_models/single_label/bioclinicalbert/{drug}_{best_model_type}.pkl",
            )


app = typer.Typer()


@app.command()
def main(drug: str = typer.Argument(help="Drug name to train models for")):
    tf_idf_single_label(drug)


if __name__ == "__main__":
    app()
