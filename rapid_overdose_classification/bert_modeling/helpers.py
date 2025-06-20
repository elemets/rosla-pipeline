import torch
import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    classification_report,
)
from transformers import EvalPrediction
from rapid_overdose_classification.bert_modeling.constants import (
    drug_cols,
)


def multi_label_metrics(predictions, labels, threshold=0.5):
    """
    This is specifically for assessing models as multi-label classifiers.
    Takes in the predictions labels and a threshold and then compares
    the labels and the predictions in order to give metrics.
    """

    # sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # threshold to turn them into predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # compute metrics
    y_true = labels
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    roc_auc = roc_auc_score(y_true, y_pred, average="macro")
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    class_report = classification_report(
        y_true,
        y_pred,
        output_dict=False,
        target_names=drug_cols,
    )
    metrics = {"f1": f1_macro_average, "roc_auc": roc_auc, "accuracy": accuracy}
    return metrics


### Function used in huggingface trainer callback
### this enables us to log as we go and is part of the standard huggingface implementation
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


class TokenizeFunc:
    """
    Tokenizer class helper. Allows for different tokenizer definitions when
    calling the tokenize_function
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize_function(self, drug_data):

        encoded_text = self.tokenizer(
            drug_data["text"], padding="max_length", truncation=True
        )

        encoded_text["labels"] = torch.FloatTensor(
            [drug_data.get(key) for key in drug_cols]
        )

        return encoded_text
