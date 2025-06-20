"""BERT training and evaluation config"""

# Training config
batch_size = 32
train_epochs = 6
metric = "f1"
lr = 2e-5
wd = 0.01
device = "cuda"

# Evaluation config
EVAL_BATCH_SIZE = 16
MAX_LENGTH = 512
BOOTSTRAP_METRICS = ["roc_auc", "accuracy", "hamming_loss", "f1_macro"]
BOOTSTRAP_N_SAMPLES = 1000
BOOTSTRAP_NUM_RESAMPLES = 1000
BOOTSTRAP_AVERAGE = "macro"

# MLFlow experiments
EXTERNAL_EXPERIMENT_NAME = "External Dataset"
INTERNAL_EXPERIMENT_NAME = "Table 3 Results"
