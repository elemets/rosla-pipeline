import os
import glob
import pandas as pd
import sys
import argparse


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

        # Drop duplicates based on "CaseNumber," keeping the first occurrence (which now has priority based on the sort)
        dmec_combined_df = dmec_combined_df.drop_duplicates(
            subset="CaseNumber", keep="first"
        )

        # Display the shape of the DataFrame after dropping duplicates
        print(
            f"DMEC combined DataFrame shape (after dropping duplicates): {dmec_combined_df.shape}"
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

        # Concatenate UCLA DataFrames and drop duplicates in 'CaseNumber'
        ucla_combined = pd.concat(ucla_dfs, ignore_index=True)
        ucla_combined = ucla_combined.sort_values(
            by=["Any Drugs", "DeathDate"], ascending=[False, False]
        )

        # Drop duplicates based on "CaseNumber," keeping the first occurrence (which now has priority based on the sort)
        ucla_combined = ucla_combined.drop_duplicates(subset="CaseNum", keep="first")

        # Display the shape of the DataFrame after dropping duplicates
        print(
            f"UCLA combined DataFrame shape (after dropping duplicates): {ucla_combined.shape}"
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
