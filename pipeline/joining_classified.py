import os
import glob
import pandas as pd
import sys
import argparse


def coalesce_duplicate_rows(df, subset):
    """
    Collapse rows that share the same value(s) in `subset` into a single row.

    Instead of arbitrarily keeping the first duplicate, the row with the most
    non-null values is used as the base, and any remaining nulls are then
    filled in using values from the other rows in the same group.

    Parameters:
    - df (pd.DataFrame): The DataFrame to deduplicate.
    - subset (str or list): Column name(s) that identify a duplicate.

    Returns:
    - pd.DataFrame: One row per unique `subset` value, with nulls minimized.
    """
    original_columns = df.columns.tolist()

    # Rank rows by how complete they are (most non-null values first).
    # A *stable* sort preserves any prior ordering (e.g. the Any Drugs /
    # DeathDate sort) as a tie-breaker among rows that are equally complete.
    completeness = df.notna().sum(axis=1)
    df_sorted = (
        df.assign(_completeness=completeness)
        .sort_values("_completeness", ascending=False, kind="stable")
        .drop(columns="_completeness")
    )

    # groupby().first() takes the first *non-null* value in each column, so
    # the base (most complete) row's values win and any gaps are filled from
    # the remaining rows in the group.
    combined = df_sorted.groupby(
        subset, as_index=False, sort=False, dropna=False
    ).first()

    # Restore the original column order (groupby moves the key columns first).
    return combined[original_columns]


def combine_classified_csv_files(folder_path, output_file=None):
    """
    Combines all CSV files ending with '_classified.csv' in the specified folder into a single DataFrame.
    If any files contain "DMEC" in their names, these are processed first, with duplicates in 'CaseNumber' removed.

    Parameters:
    - folder_path (str): The path to the folder containing the CSV files.
    - output_file (str, optional): The path to save the combined DataFrame as a CSV file. If None, the file is not saved.

    Returns:
    - combined_df (pd.DataFrame): The combined DataFrame containing data from all CSV files.
    """
    # Find all '_classified.csv' files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*_classified.csv"))
    if not csv_files:
        print(f"No files ending with '_classified.csv' found in {folder_path}.")
        return None

    # Separate files containing "DMEC" in their names
    dmec_files = [file for file in csv_files if "DMEC" in os.path.basename(file)]
    ucla_files = [
        file
        for file in csv_files
        if "UCLA" in os.path.basename(file) and "DMEC" not in os.path.basename(file)
    ]
    other_files = [
        file
        for file in csv_files
        if "DMEC" not in os.path.basename(file) and "UCLA" not in os.path.basename(file)
    ]

    # Initialize an empty list to hold dataframes
    df_list = []

    # Combine DMEC files first, removing duplicates in 'CaseNumber'
    if dmec_files:
        dmec_dfs = []
        for file in dmec_files:
            try:
                df = pd.read_csv(file)
                if "Unnamed: 0" in df.columns:
                    df = df.drop(columns=["Unnamed: 0"])

                dmec_dfs.append(df)
                print(f"Loaded DMEC file: {file} with shape {df.shape}")
            except Exception as e:
                print(f"Error reading DMEC file {file}: {e}")

        # Concatenate DMEC DataFrames and drop duplicates in 'CaseNumber'
        dmec_combined_df = pd.concat(dmec_dfs, ignore_index=True)
        dmec_combined_df = dmec_combined_df.sort_values(
            by=["Any Drugs", "DeathDate"], ascending=[False, False]
        )

        # Collapse duplicate "CaseNumber" rows: keep the most complete row and
        # fill its remaining nulls from the other duplicate rows.
        dmec_combined_df = coalesce_duplicate_rows(
            dmec_combined_df, subset="CaseNumber"
        )

        # Display the shape of the DataFrame after collapsing duplicates
        print(
            f"DMEC combined DataFrame shape (after coalescing duplicates): {dmec_combined_df.shape}"
        )

        df_list.append(dmec_combined_df)

    if ucla_files:
        ucla_dfs = []
        for file in ucla_files:
            try:
                df = pd.read_csv(file)
                if "Unnamed: 0" in df.columns:
                    df = df.drop(columns=["Unnamed: 0"])

                ucla_dfs.append(df)
                print(f"Loaded UCLA file: {file} with shape {df.shape}")
            except Exception as e:
                print(f"Error reading UCLA file {file}: {e}")

        # Concatenate UCLA DataFrames and drop duplicates in 'CaseNum'
        ucla_combined = pd.concat(ucla_dfs, ignore_index=True)
        ucla_combined = ucla_combined.sort_values(
            by=["Any Drugs", "DeathDate"], ascending=[False, False]
        )

        # Collapse duplicate "CaseNum" rows: keep the most complete row and
        # fill its remaining nulls from the other duplicate rows.
        ucla_combined = coalesce_duplicate_rows(ucla_combined, subset="CaseNum")

        # Display the shape of the DataFrame after collapsing duplicates
        print(
            f"UCLA combined DataFrame shape (after coalescing duplicates): {ucla_combined.shape}"
        )

        df_list.append(ucla_combined)

    # Combine remaining files
    if other_files:
        for file in other_files:
            try:
                df = pd.read_csv(file)
                df_list.append(df)
                print(f"Loaded file: {file} with shape {df.shape}")
            except Exception as e:
                print(f"Error reading file {file}: {e}")

    # Concatenate all dataframes in the list
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Final combined DataFrame shape: {combined_df.shape}")

    # Save to CSV if output_file is specified
    if output_file:
        combined_df.to_csv(output_file, index=False)
        print(f"Combined DataFrame saved to {output_file}")

    return combined_df


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join classified CSV files.")
    parser.add_argument("-i", "--input", required=True, help="Input folder path.")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file.")

    args = parser.parse_args()
    folder_path = args.input
    output_file = args.output

    combined_df = combine_classified_csv_files(folder_path, output_file)