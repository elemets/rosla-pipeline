from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import sys
import torch
from helpers import (
    compute_metrics,
    TokenizeFunc,
    drug_cols,
    cols_needed,
)
from constants import (
    cols_needed,
    drug_cols,
    batch_size,
    lr,
    metric,
    wd,
    train_epochs,
    device,
)


def bert_model_train(input_data, bert_type):
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
    if bert_type == "BERT":
        bert_id = ("google-bert/bert-base-cased",)
    else:
        bert_id = "emilyalsentzer/Bio_ClinicalBERT"

    tokenizer = AutoTokenizer.from_pretrained(bert_id)
    drug_data = pd.read_pickle(f"../../data/outcomes_squashed/{input_data}")
    tokenFunc = TokenizeFunc(tokenizer)

    ### adding columns needed for comparing against single label later
    cols_for_split = cols_needed.copy()
    cols_for_split.extend(["clinBERTEmbed"])
    print(cols_for_split)

    drug_data_for_bert = drug_data[cols_for_split]

    train_val, test = train_test_split(
        drug_data_for_bert, random_state=42, test_size=0.2
    )
    train, val = train_test_split(train_val, random_state=42, test_size=0.2)

    test.to_pickle("../../data/test_set.pkl")
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
        num_labels=11,
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
    )

    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    model_path = f"../../models/{model_type}/"

    # Train, evaluate, and save the model
    trainer.train()
    print("Evaluating Model:")
    evaluation = trainer.evaluate()
    print(evaluation)
    print(f"Saving trained and evaluated model to: {model_path}")
    trainer.save_model(model_path)


if __name__ == "__main__":
    input_location = sys.argv[1]
    model_type = sys.argv[2]
    bert_model_train(input_location, model_type)
