import pandas as pd
import pdfplumber
import sys
from tqdm import tqdm
import argparse


def pdf_to_csv(input_location, output_loc):
    pdf = pdfplumber.open(input_location)
    table_page_1 = pdf.pages[0].extract_table()
    cols = table_page_1[0]
    total_df = pd.DataFrame(columns=cols)
    print("Extracting each table from every page in the pdf")
    for pages in tqdm(pdf.pages):
        curr_page = pages.extract_table()
        curr_page_df = pd.DataFrame(curr_page[1:], columns=cols)
        total_df = pd.concat([total_df, curr_page_df], ignore_index=True)

    total_df.to_csv(output_loc, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDF to CSV.")
    parser.add_argument("-i", "--input", required=True, help="Input PDF file.")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file.")
    args = parser.parse_args()

    pdf_to_csv(args.input, args.output)
