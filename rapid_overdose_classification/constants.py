"""Evaluating single as multi constants"""

TEST_SET_PATH = "../../data/test_set.pkl"
EMBEDDING_TYPES = ["bioclinicalbert", "cuis", "glove"]
MODEL_PATH_TEMPLATE = "../../models/classic_ml_models/single_label/{embedding_type}/"


"""Training constants"""

PROCESSED_DATA_TFIDF_DIR = "../../data/processed_data/PROCESSED_DATA_tfidf.pkl"
PROCESSED_DATA_GLOVE_DIR = "../../data/processed_data/PROCESSED_DATA_glove.pkl"
PROCESSED_DATA_CUI_DIR = "../../data/processed_data/PROCESSED_DATA_cui.pkl"
PROCESSED_DATA_BIOCLINICALBERT_DIR = (
    "../../data/processed_data/PROCESSED_DATA_bioclinicalbert.pkl"
)
COMBINED_DATA_DIR = "../../data/processed_data/combined_data.pkl"
MULTI_LABEL_MODEL_TYPES = ["RandomForest", "XGBoost"]

model_list = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "SVM",
]


other_cols_to_squash = [
    "Anticonvulsant",
    "Antihistamine",
    "Anti-psychotic",
    "MDA",
    "MDMA",
    "Anti-Depressant",
    "Muscle Relaxants",
    "Barbiturates",
    "Hallucinogens",
    "Amphetamine",
]

benzo_cols_to_squash = ["Xanax", "Flualprazolam"]

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
]

drug_cols_comb = [
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


drug_cols_opioids = ["Heroin", "Opioid", "Fentanyl", "Prescription.opioids"]
