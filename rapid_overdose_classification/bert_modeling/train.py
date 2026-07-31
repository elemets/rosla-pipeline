import typer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import random
import os
import json
from sklearn.metrics import f1_score
from helpers import (
    compute_metrics,
    TokenizeFunc,
)
from rapid_overdose_classification.bert_modeling.config import (
    batch_size,
    lr,
    metric,
    wd,
    train_epochs,
    device,
)
from rapid_overdose_classification.bert_modeling.constants import (
    cols_needed,
    drug_cols,
    MODEL_PATH,
    DATA_PATH,
)


app = typer.Typer()


def find_best_threshold_for_label(probs_i, true_i, step=0.01):
    thresholds = np.arange(0.0, 1.0 + step, step)
    best_thr = 0.0
    best_score = 0.0

    for t in thresholds:
        preds_i = (probs_i >= t).astype(int)
        score = f1_score(true_i, preds_i, average="binary")
        if score > best_score:
            best_score = score
            best_thr = t

    return best_thr, best_score


def bert_model_train(input_data: str, bert_type: str):
    """
    Trains a BERT model on a given dataset.

    Args:
        input_data (str): Path to the input data file (pickle format).
        bert_type (str): Type of the BERT model to be used for training ('BERT' or 'Bio_ClinicalBERT').

    Returns:
        None

    This function performs the following steps:
    1. Loads the input data from a pickle file.
    2. Selects the appropriate BERT model and tokenizer based on the specified type.
    3. Splits the data into training, validation, and test sets.
    4. Tokenizes the text data.
    5. Loads the pre-trained BERT model for sequence classification.
    6. Sets up training arguments and the Trainer.
    7. Trains the model on the training dataset.
    8. Evaluates the model on the validation dataset.
    9. Saves the trained model to a specified directory.
    """
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    # If you have a GPU:
    torch.cuda.manual_seed_all(seed_value)

    if bert_type == "BERT":
        bert_id = "google-bert/bert-base-cased"
    else:
        bert_id = "emilyalsentzer/Bio_ClinicalBERT"

    tokenizer = AutoTokenizer.from_pretrained(bert_id)
    drug_data = pd.read_pickle(f"{DATA_PATH}{input_data}")
    tokenFunc = TokenizeFunc(tokenizer)

    ### adding columns needed for comparing against single label later
    cols_for_split = cols_needed.copy()
    cols_for_split.extend(["clinBERTEmbed", "vector", "GloVE_proc"])
    print(cols_for_split)

    drug_data_for_bert = drug_data[cols_for_split]

    train_val, test = train_test_split(
        drug_data_for_bert, random_state=42, test_size=0.2
    )
    train, val = train_test_split(train_val, random_state=42, test_size=0.2)

    test.to_pickle("../../data/test_set.pkl")
    train.to_pickle("../../data/train_set.pkl")
    val.to_pickle("../../data/val_set.pkl")
    print("Saved test set to the data directory")

    ### making sure dataset has only text columns and the outcome
    train = train[cols_needed]
    val = val[cols_needed]
    test = test[cols_needed]

    train_drug_set = Dataset.from_pandas(train)
    test_drug_set = Dataset.from_pandas(test)
    val_drug_set = Dataset.from_pandas(val)

    train_ds = train_drug_set.map(
        tokenFunc.tokenize_function, remove_columns=train_drug_set.column_names
    )
    test_ds = test_drug_set.map(
        tokenFunc.tokenize_function, remove_columns=test_drug_set.column_names
    )
    val_ds = val_drug_set.map(
        tokenFunc.tokenize_function, remove_columns=val_drug_set.column_names
    )

    # for pytorch compatibility
    train_ds.set_format("torch")
    test_ds.set_format("torch")
    val_ds.set_format("torch")

    # Loading the BERT pretrained model for classification
    model = AutoModelForSequenceClassification.from_pretrained(
        bert_id,
        num_labels=10,
        problem_type="multi_label_classification",
    ).to(device)

    for param in model.parameters():
        param.data = param.data.contiguous()

    training_args = TrainingArguments(
        output_dir="../Models/BERTfine_trainargs",
        eval_strategy="epoch",
        save_strategy="epoch",
        do_train=True,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_first_step=True,
        metric_for_best_model=metric,
        num_train_epochs=train_epochs,
        weight_decay=wd,
        load_best_model_at_end=True,
        seed=42,
    )

    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train, evaluate, and save the model
    trainer.train()

    # Finding best threshold
    val_output = trainer.predict(val_ds)
    val_logits = val_output.predictions
    val_labels = val_output.label_ids
    val_probs = torch.sigmoid(torch.tensor(val_logits)).numpy()
    best_thresholds = {}
    for index, drug_col in enumerate(drug_cols):
        thr, score = find_best_threshold_for_label(
            val_probs[:, index], val_labels[:, index]
        )
        best_thresholds[drug_col] = thr
        print(f"best threshold for: {drug_col} is {thr} with score {score}")

    print(best_thresholds)
    print("Evaluating Model:")
    evaluation = trainer.evaluate()
    print(evaluation)
    print(f"Saving trained and evaluated model to: {MODEL_PATH}/{bert_type}/")
    trainer.save_model(f"{MODEL_PATH}/{bert_type}/")

    thresholds_path = os.path.join(f"{MODEL_PATH}/{bert_type}/", "best_thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump(best_thresholds, f, indent=4)


@app.command()
def train(
    input_data: str = typer.Argument(
        '../../data/processed_data/combined_data_removing_mislabels_corrected.pkl', help="Path to the input data file (pickle format)"
    ),
    bert_type: str = typer.Argument(
        'Bio_ClinicalBERT', help="Type of BERT model ('BERT' or 'Bio_ClinicalBERT')"
    ),
):
    """
    Train a BERT model for drug overdose classification.
    """
    bert_model_train(input_data, bert_type)


if __name__ == "__main__":
    app()
