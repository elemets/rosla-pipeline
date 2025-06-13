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
