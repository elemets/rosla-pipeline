import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()

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

drug_cols_no_opioids = [
    "Methamphetamine",
    "Cocaine",
    "Alcohol",
    "Prescription.opioids",
    "Benzodiazepines",
    "Others",
]

drug_cols_opioids = ["Opioid", "Fentanyl", "Prescription.opioids"]


def prepping_outcome_cols(input_data):
    """
    Preprocess the input data by dropping and squashing specified columns, and creating new custom columns for analysis.

    Args:
        input_data (pd.DataFrame): DataFrame containing the input data.

    Returns:
        pd.DataFrame: DataFrame with new custom columns 'Any Drugs' and 'Drug No Opioids'.
    """
    print("Dropping and squashing others")
    others_df = input_data.progress_apply(set_others, axis=1)
    others_df = others_df.drop(columns=other_cols_to_squash)
    print("Dropping and squashing benzos")
    benzos_df = others_df.progress_apply(set_benzos, axis=1)
    benzos_df = benzos_df.drop(columns=benzo_cols_to_squash)
    benzos_df = benzos_df.progress_apply(set_any_opioids, axis=1)
    benzos_df = benzos_df.drop(columns=["Opioid"])

    print("Creating any drugs and no opioids cols")
    benzos_df["Any Drugs"] = np.zeros
    benzos_df["Drug No Opioids"] = np.zeros
    any_drugs_df = benzos_df.progress_apply(set_any_drugs, axis=1)
    drugs_no_opioids_df = any_drugs_df.progress_apply(set_drug_no_opioids, axis=1)
    return drugs_no_opioids_df


def set_others(row):
    """
    Set the 'Others' column to 1 if any of the specified columns contain 1.

    Args:
        row (pd.Series): Row of data to process.

    Returns:
        pd.Series: Updated row with 'Others' column set to 1 if applicable.
    """
    if (row[other_cols_to_squash] == 1).any():
        row["Others"] = 1
    return row


def set_benzos(row):
    """
    Set the 'Benzodiazepines' column to 1 if any of the specified columns contain 1.

    Args:
        row (pd.Series): Row of data to process.

    Returns:
        pd.Series: Updated row with 'Benzodiazepines' column set to 1 if applicable.
    """
    if (row[benzo_cols_to_squash] == 1).any():
        row["Benzodiazepines"] = 1
    return row


def set_any_drugs(row):
    """
    Set the 'Any Drugs' column to 1 if any of the specified columns contain 1.

    Args:
        row (pd.Series): Row of data to process.

    Returns:
        pd.Series: Updated row with 'Any Drugs' column set to 1 if applicable.
    """
    if (row[drug_cols] == 1).any():
        row["Any Drugs"] = 1
    else:
        row["Any Drugs"] = 0
    return row


def set_drug_no_opioids(row):
    """
    Set the 'Drug No Opioids' column to 1 if any of the specified columns contain 1.

    Args:
        row (pd.Series): Row of data to process.

    Returns:
        pd.Series: Updated row with 'Drug No Opioids' column set to 1 if applicable.
    """
    if (row[drug_cols_no_opioids] == 1).any():
        row["Drug No Opioids"] = 1
    else:
        row["Drug No Opioids"] = 0
    return row


def set_any_opioids(row):
    """
    Set the 'Any opioids' column to 1 if any of the specified columns contain 1.

    Args:
        row (pd.Series): Row of data to process.

    Returns:
        pd.Series: Updated row with 'Drug No Opioids' column set to 1 if applicable.
    """
    if (row[drug_cols_opioids] == 1).any():
        row["Any Opioids"] = 1
    else:
        row["Any Opioids"] = 0
    return row


if __name__ == "__main__":
    input_loc = sys.argv[1]
    embedding = sys.argv[2]
    input_df = pd.read_pickle(f"../data/different_embeddings/{input_loc}")
    cols_squished_df = prepping_outcome_cols(input_df)
    cols_squished_df.to_pickle(
        f"../data/outcomes_squashed/outcomes_squashed_{embedding}.pkl"
    )
