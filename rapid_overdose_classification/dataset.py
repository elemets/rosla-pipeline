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


def prepping_outcome_cols(input_data):
    """
    Preprocess the input data by dropping and squashing specified columns, and creating new custom columns for analysis.

    Args:
        input_data (pd.DataFrame): DataFrame containing the input data.

    Returns:
        pd.DataFrame: DataFrame with new custom columns 'Any Drugs'.
    """
    print("Dropping and squashing others")
    others_df = input_data.progress_apply(set_others, axis=1)
    others_df = others_df.drop(columns=other_cols_to_squash)
    print("Dropping and squashing benzos")
    benzos_df = others_df.progress_apply(set_benzos, axis=1)
    benzos_df = benzos_df.drop(columns=benzo_cols_to_squash)
    benzos_df = benzos_df.progress_apply(set_any_opioids, axis=1)
    benzos_df = benzos_df.drop(columns=["Opioid"])

    print("Creating any drugs column")
    benzos_df["Any Drugs"] = np.zeros
    any_drugs_df = benzos_df.progress_apply(set_any_drugs, axis=1)
    return any_drugs_df


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
    else:
        row["Others"] = 0
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


def set_any_opioids(row):
    """
    Set the 'Any opioids' column to 1 if any of the specified columns contain 1.

    Args:
        row (pd.Series): Row of data to process.

    Returns:
        pd.Series: Updated row with 'Any Opioids' column set to 1 if applicable.
    """
    if (row[drug_cols_opioids] == 1).any():
        row["Any Opioids"] = 1
    else:
        row["Any Opioids"] = 0
    return row


def combine_embedding_files():
    cui = pd.read_pickle(f"../data/outcomes_squashed/outcomes_squashed_cui.pkl")
    bioclin = pd.read_pickle(
        f"../data/outcomes_squashed/outcomes_squashed_bioclinicalbert.pkl"
    )
    glove = pd.read_pickle(f"../data/outcomes_squashed/outcomes_squashed_glove.pkl")
    combined_df = pd.DataFrame()
    combined_df["text"] = cui["text"]
    combined_df["vector"] = cui["vector"]
    combined_df["clinBERTEmbed"] = bioclin["clinBERTEmbed"]
    combined_df["GloVE_proc"] = glove["GloVE_proc"]
    combined_df[drug_cols_comb] = bioclin[drug_cols_comb]
    print(combined_df.columns.tolist())
    print("Saving combined pkl file")
    return combined_df


if __name__ == "__main__":
    input_loc = sys.argv[1]

    if input_loc == "combine":
        comb_df = combine_embedding_files()
        comb_df.to_pickle("../data/outcomes_squashed/combined_data.pkl")
    else:
        embedding = sys.argv[2]
        input_df = pd.read_pickle(f"../data/different_embeddings/{input_loc}")
        cols_squished_df = prepping_outcome_cols(input_df)
        cols_squished_df.to_pickle(
            f"../data/outcomes_squashed/outcomes_squashed_{embedding}.pkl"
        )
