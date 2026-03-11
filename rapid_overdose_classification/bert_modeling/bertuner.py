from light_tuner.bertuner.BERTuner import BERTuneClassifier
import pandas as pd

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

drugs_df = pd.read_pickle(f"/home/afunnell/Code/Rapid_overdose_clean/data/processed_data/combined_data_removing_mislabels.pkl")


SEED = 42
DEFAULT_MODEL_CHOICES = {
    "bioclinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "roberta-base": "roberta-base",
    "distilbert": "distilbert-base-uncased",
    "bert-base": "google-bert/bert-base-cased",
    "electra-small": "google/electra-small-discriminator",
    "electra-base": "google/electra-base-discriminator",
}

MULTILABEL_SEARCHSPACE = {
    "model": ["bioclinicalbert", "bert-base"],
    "learning_rate": {"low": 1e-6, "high": 5e-5, "log": True},
    "batch_size": [16, 32],
    "weight_decay": {"low": 0.0, "high": 0.05},
    "warmup_ratio": {"low": 0.0, "high": 0.1},
    "scheduler": ["linear"],
    "dropout": {"low": 0.0, "high": 0.2},
    "early_stopping_patience": {"low": 2, "high": 4},
}

classifier_opt = BERTuneClassifier(
    dataframe=drugs_df,
    models_dir="./models/BERTModels",
    text_feature="text",
    target_cols=drug_cols, 
    log_level="verbose"

)
## initializing model choices with default options
classifier_opt.initialize_model_choices()
## initializing search space with default options
classifier_opt.initialize_search_space(MULTILABEL_SEARCHSPACE)
## Optimizing will find the best parameters and save them to the model object
classifier_opt.optimize(
    n_trials=2,
    optimize_metric="f1_macro",  
    study_name="bertclass_multilabel_test",
    greater_is_better=True
)
## Here we train the model on these best parameters
classifier_opt.train_final_model(run_name="Finetune Drug Run")
