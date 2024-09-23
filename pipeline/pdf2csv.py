import pandas as pd
import pdfplumber
import sys
from tqdm import tqdm


def pdf_to_csv(input_location):
    pdf = pdfplumber.open(input_location)
    table_page_1 = pdf.pages[0].extract_table()
    cols = table_page_1[0]
    total_df = pd.DataFrame(columns=cols)
    print("Extracting each table from every page in the pdf")
    for pages in tqdm(pdf.pages):
        curr_page = pages.extract_table()
        curr_page_df = pd.DataFrame(curr_page[1:], columns=cols)
        total_df = pd.concat([total_df, curr_page_df], ignore_index=True)

    output_loc = input_location[:-4] + ".csv"
    total_df.to_csv(output_loc, index=False)


if __name__ == "__main__":
    input_location = sys.argv[1]
    pdf_to_csv(input_location)
