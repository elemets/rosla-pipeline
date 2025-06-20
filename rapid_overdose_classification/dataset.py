import pandas as pd
import numpy as np
from tqdm import tqdm
import typer
from rapid_overdose_classification.constants import (
    drug_cols_comb,
    drug_cols_opioids,
    drug_cols,
    benzo_cols_to_squash,
    other_cols_to_squash,
)

tqdm.pandas()

app = typer.Typer(help="Preprocessing outcome columns and combining embedding files")


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
    """
    Combine embedding files from different methods into a single dataframe.

    Returns:
        pd.DataFrame: Combined dataframe with all embeddings and outcomes.
    """
    print("Loading embedding files...")
    cui = pd.read_pickle("../data/processed_data/processed_data_cui.pkl")
    bioclin = pd.read_pickle(
        "../data/processed_data/processed_data_bioclinicalbert.pkl"
    )
    glove = pd.read_pickle("../data/processed_data/processed_data_glove.pkl")

    print("Combining embeddings...")
    combined_df = pd.DataFrame()
    combined_df["text"] = cui["text"]
    combined_df["vector"] = cui["vector"]
    combined_df["clinBERTEmbed"] = bioclin["clinBERTEmbed"]
    combined_df["GloVE_proc"] = glove["GloVE_proc"]
    combined_df[drug_cols_comb] = bioclin[drug_cols_comb]

    print("Combined dataframe columns:", combined_df.columns.tolist())
    return combined_df


@app.command()
def process(
    input_file: str = typer.Argument(
        help="Input pickle file name (without path, e.g., 'cui_vec_test.pkl')"
    ),
    embedding_type: str = typer.Argument(
        help="Type of embedding (e.g., 'cui', 'bioclinicalbert', 'glove')"
    ),
    input_dir: str = typer.Option(
        "../data/different_embeddings/", help="Directory containing input files"
    ),
    output_dir: str = typer.Option(
        "../data/processed_data/", help="Directory to save output files"
    ),
):
    """
    Process a single embedding file by preprocessing outcome columns.
    """
    input_path = f"{input_dir}{input_file}"
    output_path = f"{output_dir}processed_data_{embedding_type}.pkl"

    print(f"Loading data from: {input_path}")
    try:
        input_df = pd.read_pickle(input_path)
        print(f"Loaded dataframe with shape: {input_df.shape}")
    except FileNotFoundError:
        typer.echo(f"Error: File {input_path} not found.", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error loading file: {e}", err=True)
        raise typer.Exit(1)

    print("Processing outcome columns...")
    cols_squished_df = prepping_outcome_cols(input_df)

    print(f"Saving processed data to: {output_path}")
    cols_squished_df.to_pickle(output_path)
    print(f"Successfully processed {embedding_type} embeddings!")
    print(f"Output shape: {cols_squished_df.shape}")


@app.command()
def combine(
    input_dir: str = typer.Option(
        "../data/processed_data/",
        help="Directory containing embedding files to combine",
    ),
    output_dir: str = typer.Option(
        "../data/processed_data/", help="Directory to save combined file"
    ),
    output_filename: str = typer.Option(
        "combined_data.pkl", help="Name of the output combined file"
    ),
):
    """
    Combine multiple embedding files into a single dataframe.
    """
    print("Starting combination process...")

    try:
        comb_df = combine_embedding_files()
        output_path = f"{output_dir}{output_filename}"

        print(f"Saving combined file to: {output_path}")
        comb_df.to_pickle(output_path)
        print(f"Successfully saved combined dataframe!")
        print(f"Combined dataframe shape: {comb_df.shape}")
        print(f"Final columns: {comb_df.columns.tolist()}")

    except FileNotFoundError as e:
        typer.echo(f"Error: Required embedding file not found: {e}", err=True)
        typer.echo(
            "Make sure you have run the process command for cui, bioclinicalbert, and glove embeddings first.",
            err=True,
        )
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error during combination: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def info():
    """
    Display information about available drug columns and preprocessing steps.
    """
    print("Drug Columns Information:")
    print("=" * 50)
    print(f"Main drug columns: {drug_cols}")
    print(f"Opioid columns: {drug_cols_opioids}")
    print(f"Combined drug columns: {drug_cols_comb}")
    print(f"Benzo columns to squash: {benzo_cols_to_squash}")
    print(f"Other columns to squash: {other_cols_to_squash}")
    print("\nProcessing Steps:")
    print("1. Squash 'Others' columns into single 'Others' column")
    print("2. Squash 'Benzodiazepines' columns into single 'Benzodiazepines' column")
    print("3. Create 'Any Opioids' column")
    print("4. Create 'Any Drugs' column")


if __name__ == "__main__":
    app()
