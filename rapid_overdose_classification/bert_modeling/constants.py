MODEL_PATH = "../../models/bert_models/"
DATA_PATH = "../../data/processed_data/"
TEST_SET_PATH = "../../data/test_set.pkl"
TRAINING_OUTPUT_DIR = "../Models/BERTfine_trainargs"
REPORTS_OUTPUT_PATH = "../../reports/model_outputs/"
EXPLAINABILITY_OUTPUT_PATH = "../../reports/bioclinicalBERT_explainability/"

EVALUATION_RESULTS_PATH = (
    "../../reports/evaluated_res_external_removedmislabels_n_model.csv"
)
EVALUATION_MISMATCHES_PATH = (
    "../../reports/predicted_wrong_external_removedmislabels_n_model.csv"
)
EVALUATION_METRICS_PATH = (
    "../../reports/eval_metric_external_removedmislabels_n_model.csv"
)

### Defining the columns for the classification report
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
]


cols_needed = [
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
    "text",
]
