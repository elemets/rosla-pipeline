from constants import all_drug_cols, model_list
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from model_tuner import Model, loadObjects, dumpObjects
import sklearnex
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.base import clone
from NaiveSVC import NaivelyCalibratedLinearSVC
import sys
from tqdm import tqdm


def bioclinicalbert_single_label(drug):
    ### this patch helps to speed up sklearn in general
    ### but especially SVC which is veeeery slow due
    ### to using the probabiltiy=True (which is needed to generate roc_auc etc.)
    sklearnex.patch_sklearn()
    drug_df = pd.read_pickle(
        "../../data/outcomes_squashed/outcomes_squashed_bioclinicalbert.pkl"
    )

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    """
    Repeat the same steps but using the bioclinicalBERT embeddings.
    The drug_df['clinBERTEmbed'] column contains the bioclinicalBERT embeddings.
    Log to a different location on mlflow. 
    """

    experiment_name = f"bioclinicalBERT Bootstrapped"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{drug}") as parent_run:

        best_average_precision = 0
        best_model = 0

        for model_name in tqdm(model_list):

            with mlflow.start_run(run_name=f"{model_name}", nested=True) as child_run:

                ## Splitting data and calculating ratio for scale_pos_weight
                y = drug_df[drug].values
                X = drug_df["clinBERTEmbed"].values
                X = np.stack(X, axis=0)
                n_samples, sequence_length, n_features = X.shape
                X = X.reshape(-1, n_features)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                positive_count = np.sum(y)
                negative_count = len(y) - positive_count
                scale_pos_weight = negative_count / positive_count

                if model_name == "Random Forest":
                    estimator = RandomForestClassifier(
                        class_weight="balanced", n_jobs=-2
                    )

                    estimator_name = "rf"

                    tuned_parameters = {
                        f"{estimator_name}__max_depth": [3, 5, 10],
                        f"{estimator_name}__n_estimators": [10, 100],
                        f"{estimator_name}__max_features": [1, 3, 5, 7],
                        f"{estimator_name}__min_samples_leaf": [1, 2, 3],
                    }
                elif model_name == "Logistic Regression":

                    estimator = LogisticRegression(
                        class_weight="balanced", C=1, max_iter=2000, n_jobs=-2
                    )

                    estimator_name = "lg"
                    # Set the parameters by cross-validation
                    tuned_parameters = [{estimator_name + "__C": np.logspace(-4, 0, 3)}]
                elif model_name == "XGBoost":

                    estimator = XGBClassifier(
                        objective="binary:logistic",
                        scale_pos_weight=scale_pos_weight,
                    )

                    estimator_name = "xgb"

                    tuned_parameters = {
                        f"{estimator_name}__max_depth": [3, 5, 10, 15],
                        f"{estimator_name}__learning_rate": [0.03, 0.003, 0.001],
                        f"{estimator_name}__n_estimators": [50, 10, 100, 200],
                        f"{estimator_name}__n_jobs": [-2],
                    }
                elif model_name == "SVM":

                    estimator = NaivelyCalibratedLinearSVC(class_weight="balanced")
                    estimator_name = "svm"

                    tuned_parameters = {
                        f"{estimator_name}__tol": [0.0001, 0.03, 0.003],
                        f"{estimator_name}__C": [1, 0.05, 0.5, 0.1],
                    }

                kfold = True
                calibrate = False

                model = Model(
                    name=f"{model_name}",
                    estimator_name=estimator_name,
                    model_type="classification",
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

                model.grid_search_param_tuning(X_train, y_train, f1_beta_tune=True)

                model.fit(X_train, y_train)

                model.return_metrics(X_train, y_train, True)

                ### Logging the validation results to MLFflow
                classreport = model.classification_report

                mlflow.log_metric(
                    "f1_score_valid", classreport["macro avg"]["f1-score"]
                )
                mlflow.log_metric(
                    "precision_valid", classreport["macro avg"]["precision"]
                )
                mlflow.log_metric("recall_valid", classreport["macro avg"]["recall"])

                model.kfold = False
                print(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                X_test = pd.DataFrame(X_test)
                y_test = pd.Series(y_test)

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

                y_pred = model.predict(X_test, optimal_threshold=True)

                cm = confusion_matrix(y_test, y_pred)
                cm_display = ConfusionMatrixDisplay(cm)
                cm_display.plot()
                plt.title(f"Confusion Matrix for: {drug} on test set")
                mlflow.log_figure(cm_display.figure_, f"confusion matrix {drug}.png")

                average_precision_metric = next(
                    (
                        metric
                        for metric in bootstrap_metrics_dict
                        if metric["Metric"] == "average_precision"
                    ),
                    None,
                )
                print("AVERAGE PRECISION METRICS")
                print(average_precision_metric)

                if average_precision_metric["Mean"] > best_average_precision:
                    best_average_precision = average_precision_metric["Mean"]
                    best_model = model
                    best_model_type = model_name

        dumpObjects(
            best_model,
            f"../../models/classic_ml_models/single_label/bioclinicalbert/{drug}_{best_model_type}.pkl",
        )


if __name__ == "__main__":
    drug = sys.argv[1]
    bioclinicalbert_single_label(drug)
