# Rapid Overdose Classification and Pipeline

This repository contains two related parts:

- `pipeline/`: the updated data processing pipeline for converting raw files,
  classifying overdose records, applying regex corrections, geocoding records,
  and producing final grouped outputs. To run the pipeline or read the current
  operating instructions, see `pipeline/README.md`.
- `rapid_overdose_classification/`: the model development code and steps used
  to train the BERT/BioClinicalBERT overdose classification models.

## Classification Only

To classify text without running the full pipeline, use `classify_only.py` from
the repository root. It runs the BERT model and the NFLIS regex layer and
returns one row per record with the drug label columns.

Setup — install the requirements and put a trained checkpoint in
`models/bert_models/<name>/`:

```bash
make install          # or: pip install -r requirements.txt
```

The trained model is hosted on Hugging Face at
[Elemets/DrugDetectionBERT](https://huggingface.co/Elemets/DrugDetectionBERT).
Download it into the location the code expects:

```bash
pip install -U "huggingface_hub[cli]"
hf download Elemets/DrugDetectionBERT --local-dir models/bert_models/bioclinicalbert
```

Classify a single string, or a `.csv`/`.xlsx` file:

```bash
make classify TEXT="ACUTE COMBINED FENTANYL AND METHAMPHETAMINE TOXICITY"
make classify INPUT=records.csv OUTPUT=classified.csv

python classify_only.py "ACUTE COMBINED FENTANYL AND METHAMPHETAMINE TOXICITY"
python classify_only.py --input records.csv --output classified.csv
```

Input files are read from the coroner cause-of-death columns (`CauseA`-`CauseD`,
`CauseOther`, `HowInjuryOccurred`, `InjuryDesc`) or from a single `text` column.

From Python, one function takes a string, a list of strings, a file path, or a
DataFrame:

```python
from classify_only import classify

classify("ACUTE HEROIN TOXICITY")
classify("records.csv", output="classified.csv")
```

The checkpoint defaults to `models/bert_models/bioclinicalbert`; override it with
`MODEL=` (make), `--model` (CLI), `model=` (Python), or the `ROC_MODEL`
environment variable. Add `--skip-regex` / `use_regex=False` for BERT alone.

## Model

| | |
| --- | --- |
| Hugging Face | https://huggingface.co/Elemets/DrugDetectionBERT |
| Base model | `emilyalsentzer/Bio_ClinicalBERT` |
| Task | Multi-label classification, 10 drug labels |

Outputs are sigmoid scores. Use the per-label cut-offs in `best_thresholds.json`
that ship with the checkpoint rather than a flat 0.5.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Elemets/DrugDetectionBERT")
model = AutoModelForSequenceClassification.from_pretrained("Elemets/DrugDetectionBERT")
```

Full details of the BERT model training and validation process are described in
the Journal of Forensic Sciences paper:

Funnell, A. J., Petousis, P., Harel-Canada, F., Romero, R., Bui, A. A. T.,
Koncsol, A., Chaturvedi, H., Shover, C., & Goodman-Meza, D. (2026). Improving
drug identification in overdose death surveillance by using clinical natural
language processing models. *Journal of Forensic Sciences*. Advance online
publication. https://doi.org/10.1111/1556-4029.70281

Paper PDF:
https://onlinelibrary.wiley.com/doi/pdf/10.1111/1556-4029.70281
