#!/usr/bin/env python
import sys
import os
import random
import json
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
)


def lm_tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokenizes a batch of examples for causal language modeling.
    Assumes that each example has a "text" field.
    """
    return tokenizer(examples["text"], truncation=True, max_length=max_length)


def small_lm_train(input_data, model_type):
    """
    Finetunes a small causal language model (e.g., SmolLM2) on a given text dataset.

    Args:
        input_data (str): Path to the input pickle file, which should contain a DataFrame
                          having at least one column "text".
        model_type (str): Use "SmolLM2" to fine-tune the SmolLM2-135M-Instruct model.
    """
    # Set a fixed seed for reproducibility
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    # Select model based on the provided model_type argument
    if model_type == "SmolLM2":
        model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    else:
        print("Unsupported model type for LM training. Use 'SmolLM2'.")
        sys.exit(1)

    # Load tokenizer from the selected model
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load the data from a pickle file; expecting a DataFrame with a "text" column.
    try:
        data = pd.read_pickle(input_data)
    except Exception as e:
        print(f"Error loading data from {input_data}: {e}")
        sys.exit(1)

    if "text" not in data.columns:
        print("Error: Input data must contain a 'text' column.")
        sys.exit(1)

    data = data.drop(columns=["vector", "clinBERTEmbed", "GloVE_proc"])

    # Optionally, filter out any rows with empty or whitespace-only texts.
    data = data[data["text"].str.strip() != ""]

    # Split the DataFrame into train, validation, and test subsets. (80/10/10 split)
    train_val, test = train_test_split(data, random_state=seed_value, test_size=0.2)
    train, val = train_test_split(train_val, random_state=seed_value, test_size=0.2)

    test_path = "test_set.pkl"
    test.to_pickle(test_path)
    print(f"Saved test set to {test_path}")

    train_dataset = Dataset.from_pandas(train)
    val_dataset = Dataset.from_pandas(val)

    max_length = 512  
    tokenize_fn = lambda examples: lm_tokenize_function(examples, tokenizer, max_length=max_length)
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=val_dataset.column_names)

    



    # Set dataset format to PyTorch tensors.
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    # Load the causal language model.
    model = AutoModelForCausalLM.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Use a data collator specifically for causal language modeling.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Define training arguments.
    training_args = TrainingArguments(
        output_dir="./models/SmolLM2_finetuned",  # Directory to save model and checkpoints
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=100,
        seed=seed_value,
    )

    # NOTE: The 'tokenizer' parameter here triggers a deprecation warning.
    # To avoid it, you may remove the `tokenizer=tokenizer` argument, but be sure
    # that your data collator or training procedure does not rely on it.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,  # Remove this if you wish to avoid the deprecation warning.
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    print("Starting training...")
    trainer.train()

    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    trainer.save_model(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train.py <input_data_pickle> <model_type>")
        print("Example: python train.py data/my_data.pkl SmolLM2")
        sys.exit(1)

    input_location = sys.argv[1]
    model_type = sys.argv[2]
    small_lm_train(input_location, model_type)
