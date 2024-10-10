import os
import glob
import pandas as pd
import sys


def combine_classified_csv_files(folder_path, output_file=None):
    """
    Combines all CSV files ending with '_classified.csv' in the specified folder into a single DataFrame.

    Parameters:
    - folder_path (str): The path to the folder containing the CSV files.
    - output_file (str, optional): The path to save the combined DataFrame as a CSV file. If None, the file is not saved.

    Returns:
    - combined_df (pd.DataFrame): The combined DataFrame containing data from all CSV files.
    """
    # Use glob to find all files ending with '_classified.csv' in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*classified.csv"))

    # Check if any files were found
    if not csv_files:
        print(f"No files ending with 'classified.csv' found in {folder_path}.")
        return None

    # Initialize an empty list to hold dataframes
    df_list = []

    # Iterate over the list of files and read each one
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
            print(f"Loaded file: {file} with shape {df.shape}")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Concatenate all dataframes in the list
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined DataFrame shape: {combined_df.shape}")

    # Save to CSV if output_file is specified
    if output_file:
        combined_df.to_csv(output_file, index=False)
        print(f"Combined DataFrame saved to {output_file}")

    return combined_df


# Example usage:
if __name__ == "__main__":
    folder_path = sys.argv[1]  # Replace with your folder path
    output_file = (
        "combined_classified_data.csv"  # Replace with your desired output file name
    )

    combined_df = combine_classified_csv_files(folder_path, output_file)
