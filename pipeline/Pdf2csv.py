import sys
import tabula
import pandas as pd
from tqdm import tqdm


def pdf2csv(input_file, output_file):
    """
    Function to read in PDF files and convert the tables in them into .CSVs

    Inputs:

    input_file (str): Path to the input file (should be a .pdf)

    output_file (str): User specified path, determines where the function outputs.

    Returns None

    Notes:
    Saves a .csv at the output_file path.
    """
    print("Reading in tables:")
    list_of_tables = tabula.read_pdf(input_file, pages="all")
    print(f"Read in {len(list_of_tables)} pages of tables now converting each to csv")

    dataframe_list = []
    print("Looping through each page of PDF and converting to dataframe:")
    for table in tqdm(list_of_tables):
        ## Fill the case nums where there are NaNs using forward fill to
        ## fill until another case num is encountered
        ## This allows us to merge them into one row (when we read in each row is
        ## separated by new lines)
        table["CaseNum_filled"] = table["CaseNum"].fillna(method="ffill")
        df_merged = table.groupby("CaseNum_filled").agg(
            lambda x: " ".join(str(v) for v in x if pd.notnull(v))
        )
        df_merged = df_merged.reset_index(drop=True)
        ## Split Age and Gender into two separate columns as they are initially read
        ## in as one column
        try:
            df_merged[["Age", "Gender"]] = df_merged["Age Gender"].str.extract(
                r"(\d+\s*(?:Years|Months))\s*(\w+)"
            )

            ## Drop the old 'Age Gender' column
            df_merged = df_merged.drop(
                columns=["Unnamed: 0", "Unnamed: 1", "Age Gender"]
            )
        except:
            pass
        dataframe_list.append(df_merged)

    final_dataframe = pd.concat(dataframe_list, ignore_index=True)
    print("Outputting final dataframe")
    final_dataframe.to_csv(output_file)


if __name__ == "__main__":

    input_file = sys.argv[1]
    output_loc = sys.argv[2]

    pdf2csv(input_file, output_loc)
