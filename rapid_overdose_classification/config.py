from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from rapid_overdose_classification.classic_ml_modeling.NaiveSVC import (
    NaivelyCalibratedLinearSVC,
)
import numpy as np
from sklearn.multioutput import MultiOutputClassifier

MLFLOW_URI = "http://127.0.0.1:5000"


"""Model training configurations MULTI LABEL"""

# Multi-label specific parameters
MULTI_LABEL_BOOTSTRAP_METRICS = ["roc_auc", "accuracy", "hamming_loss", "f1_macro"]
MULTI_LABEL_N_ITER = 3  # Fewer iterations for multi-label due to complexity

multi_label_model_parameters = {
    "XGBoost": {
        "estimator": MultiOutputClassifier(XGBClassifier()),
        "xgbearly": False,
        "estimator_name": "xgb",
        "tuned_parameters": {
            "xgb__estimator__max_depth": [3, 5, 10],
            "xgb__estimator__learning_rate": [0.03, 0.003],
            "xgb__estimator__n_estimators": [50, 10, 100],
            "xgb__estimator__n_jobs": [-2],
            "xgb__estimator__device": ["cuda"],
        },
    },
    "RandomForest": {
        "estimator": MultiOutputClassifier(
            RandomForestClassifier(class_weight="balanced")
        ),
        "xgbearly": False,
        "estimator_name": "rf",
        "tuned_parameters": {
            "rf__estimator__max_depth": [3, 5, 10, None],
            "rf__estimator__n_estimators": [10, 100, 200],
            "rf__estimator__max_features": [1, 3, 5, 7],
            "rf__estimator__min_samples_leaf": [1, 2, 3],
        },
    },
}


"""Model training configurations SINGLE LABEL"""

BOOTSTRAP_METRICS = ["f1_macro", "roc_auc", "average_precision"]
STRATIFY_Y = True
RANDOM_GRID = True
KFOLD = True
N_SPLITS = 10
N_ITER = 10
MODEL_SCORING_METRIC = "roc_auc"

BOOTSTRAP_NUM_RESAMPLES = 1000
BOOTSTRAP_NUM_SAMPLES = 1000
OPTIMAL_THRESH = True
RANDOM_STATE = 42

model_parameters = {
    "Random Forest": {
        "estimator": RandomForestClassifier(class_weight="balanced"),
        "xgbearly": False,
        "estimator_name": "rf",
        "tuned_parameters": {
            "rf__max_depth": [3, 5, 10, None],
            "rf__n_estimators": [10, 100, 200],
            "rf__max_features": [1, 3, 5, 7],
            "rf__min_samples_leaf": [1, 2, 3],
        },
    },
    "Logistic Regression": {
        "estimator": LogisticRegression(class_weight="balanced", C=1, max_iter=1000),
        "xgbearly": False,
        "estimator_name": "lg",
        "tuned_parameters": {
            "lg__C": np.logspace(-4, 0, 3),
        },
    },
    "XGBoost": {
        "estimator": XGBClassifier(objective="binary:logistic"),
        "xgbearly": True,
        "estimator_name": "xgb",
        "tuned_parameters": {
            "xgb__max_depth": [3, 5, 10],
            "xgb__learning_rate": [0.03, 0.003],
            "xgb__n_estimators": [50, 10, 100],
            "xgb__n_jobs": [-2],
        },
    },
    "SVM": {
        "estimator": NaivelyCalibratedLinearSVC(class_weight="balanced"),
        "xgbearly": False,
        "estimator_name": "svm",
        "tuned_parameters": {
            "svm__tol": [0.0001, 0.03, 0.003],
            "svm__C": [1, 0.05, 0.5, 0.1],
        },
    },
}
