from constants import model_list
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
from NaiveSVC import NaivelyCalibratedLinearSVC

import sys
from tqdm import tqdm


def glove_single_label(drug):
    """
    Start GloVE embeddings experiment
    drug_df['GloVE_proc'] contains the glove embeddings.
    These will be logged to a different location on MLFlow.
    """
    experiment_name = f"GloVe Embeddings Bootstrapped"
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{drug}") as parent_run:

        best_average_precision = 0
        best_model = 0

        ### this patch helps to speed up sklearn in general
        ### but especially SVC which is veeeery slow due
        ### to using the probabiltiy=True (which is needed to generate roc_auc etc.)
        sklearnex.patch_sklearn()
        drug_df = pd.read_pickle(
            "../../data/outcomes_squashed/outcomes_squashed_glove.pkl"
        )

        for model_name in tqdm(model_list):
            with mlflow.start_run(run_name=f"{model_name}", nested=True) as child_run:

                if model_name == "Random Forest":
                    estimator = RandomForestClassifier(class_weight="balanced")

                    estimator_name = "rf"

                    tuned_parameters = {
                        f"{estimator_name}__max_depth": [3, 5, 10, None],
                        f"{estimator_name}__n_estimators": [10, 100, 200],
                        f"{estimator_name}__max_features": [1, 3, 5, 7],
                        f"{estimator_name}__min_samples_leaf": [1, 2, 3],
                    }
                elif model_name == "Logistic Regression":

                    estimator = LogisticRegression(
                        class_weight="balanced", C=1, max_iter=1000
                    )

                    estimator_name = "lg"
                    # Set the parameters by cross-validation
                    tuned_parameters = [{estimator_name + "__C": np.logspace(-4, 0, 3)}]
                elif model_name == "XGBoost":

                    estimator = XGBClassifier(
                        objective="binary:logistic",
                    )

                    estimator_name = "xgb"

                    tuned_parameters = {
                        f"{estimator_name}__max_depth": [3, 5, 10],
                        f"{estimator_name}__learning_rate": [0.03, 0.003],
                        f"{estimator_name}__n_estimators": [50, 10, 100],
                        f"{estimator_name}__n_jobs": [-2],
                    }
                elif model_name == "SVM":

                    estimator = NaivelyCalibratedLinearSVC(class_weight="balanced")
                    estimator_name = "svm"

                    tuned_parameters = {
                        f"{estimator_name}__tol": [0.0001, 0.03, 0.003],
                        f"{estimator_name}__C": [1, 0.05, 0.5, 0.1],
                    }

                y = drug_df[drug].values
                X = drug_df["GloVE_proc"].values
                X = np.stack(X, axis=0)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                kfold = True
                calibrate = False

                model = Model(
                    name=f"{model_name}",
                    estimator_name=estimator_name,
                    calibrate=calibrate,
                    estimator=clone(estimator),
                    kfold=kfold,
                    stratify_y=True,
                    grid=tuned_parameters,
                    randomized_grid=True,
                    n_iter=10,
                    scoring=["roc_auc"],
                    n_splits=10,
                    n_jobs=-2,
                    random_state=42,
                )

                print(f"Tuning hyperparameters for: {drug}")

                model.grid_search_param_tuning(X_train, y_train, f1_beta_tune=False)

                model.fit(X_train, y_train, score="roc_auc")

                model.return_metrics(X_train, y_train)

                ### Logging the validation results to MLFflow
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

                ## Using the updated model tuner class to return bootstrapped metrics
                ## For the f1 score. This is needed to recreate David's paper
                bootstrap_metrics = model.return_bootstrap_metrics(
                    X_test,
                    y_test,
                    ["f1_weighted", "roc_auc", "average_precision"],
                    num_resamples=1000,
                    n_samples=1000,
                    threshold=model.threshold["roc_auc"],
                    balance=True,
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

                y_pred = model.predict(X_test, optimal_threshold=False)

                ### Saving the confusion matrix as a plot and then logging that plot
                ### as an artifact in MLFlow
                cm = confusion_matrix(y_test, y_pred)
                cm_display = ConfusionMatrixDisplay(cm)
                cm_display.plot()
                plt.title(f"Confusion Matrix for: {drug} on test set")
                mlflow.log_figure(cm_display.figure_, f"confusion matrix {drug}.png")

                ### Logging parameters
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


if __name__ == "__main__":
    drug = sys.argv[1]
    glove_single_label(drug)
