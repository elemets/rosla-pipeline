# Rapid Overdose Classification and Pipeline

This repository contains two related parts:

- `pipeline/`: the updated data processing pipeline for converting raw files,
  classifying overdose records, applying regex corrections, geocoding records,
  and producing final grouped outputs. To run the pipeline or read the current
  operating instructions, see `pipeline/README.md`.
- `rapid_overdose_classification/`: the model development code and steps used
  to train the BERT/BioClinicalBERT overdose classification models.

Full details of the BERT model training and validation process are described in
the Journal of Forensic Sciences paper:

Funnell, A. J., Petousis, P., Harel-Canada, F., Romero, R., Bui, A. A. T.,
Koncsol, A., Chaturvedi, H., Shover, C., & Goodman-Meza, D. (2026). Improving
drug identification in overdose death surveillance by using clinical natural
language processing models. *Journal of Forensic Sciences*. Advance online
publication. https://doi.org/10.1111/1556-4029.70281

Paper PDF:
https://onlinelibrary.wiley.com/doi/pdf/10.1111/1556-4029.70281
