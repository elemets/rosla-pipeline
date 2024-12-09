import os
import glob
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

RAWDIR = "./pipeline_steps/input_files/raw/"
CONVERTEDDIR = "./pipeline_steps/input_files/converted/"
CLASSIFIEDDIR = "./pipeline_steps/input_files/classified/"
OUTPUTDIR = "./pipeline_steps/input_files"
LOGFILE = "./pipeline_steps/logs/pipeline_summary.txt"
MODEL_NAME = "bioclinicalbert"


def rename_files(directory):
    logging.info("Renaming files to remove spaces...")
    for filepath in Path(directory).glob("*"):
        if " " in filepath.name:
            new_name = filepath.name.replace(" ", "_")
            new_path = filepath.with_name(new_name)
            logging.info(f"Renaming '{filepath.name}' to '{new_name}'")
            filepath.rename(new_path)
    logging.info("File renaming completed.")


def convert_pdfs():
    logging.info("Converting PDFs to CSVs...")
    os.makedirs(CONVERTEDDIR, exist_ok=True)
    pdf_files = glob.glob(os.path.join(RAWDIR, "*.pdf"))
    for pdf_file in pdf_files:
        basename = os.path.splitext(os.path.basename(pdf_file))[0]
        csv_file = os.path.join(CONVERTEDDIR, f"{basename}.csv")
        if not os.path.exists(csv_file):
            logging.info(f"Converting '{pdf_file}' to '{csv_file}'")
            subprocess.run(
                ["python", "pdf2csv.py", "-i", pdf_file, "-o", csv_file], check=True
            )
        else:
            logging.info(f"CSV file '{csv_file}' already exists. Skipping conversion.")
    logging.info("PDF to CSV conversion completed.")


def classify_csvs():
    logging.info("Classifying CSV files...")
    os.makedirs(CLASSIFIEDDIR, exist_ok=True)
    csv_files = (
        glob.glob(os.path.join(RAWDIR, "*.csv"))
        + glob.glob(os.path.join(CONVERTEDDIR, "*.csv"))
        + glob.glob(os.path.join(RAWDIR, "*.xlsx"))
    )
    for csv_file in csv_files:
        basename = os.path.splitext(os.path.basename(csv_file))[0]
        classified_file = os.path.join(CLASSIFIEDDIR, f"{basename}_classified.csv")
        logging.debug(f"Checking if '{classified_file}' exists.")
        if not os.path.exists(classified_file):
            logging.info(f"Classifying '{csv_file}' to '{classified_file}'")
            command = [
                "python",
                "classify.py",
                "-i",
                csv_file,
                "-o",
                classified_file,
                "-m",
                MODEL_NAME,
            ]
            logging.debug(f"Running command: {' '.join(command)}")
            subprocess.run(command, check=True)
        else:
            logging.info(
                f"Classified file '{classified_file}' already exists. Skipping classification."
            )
    logging.info("CSV classification completed.")


def join_classified_data():
    logging.info("Joining classified data...")
    output_file = os.path.join(OUTPUTDIR, "combined_classified_data.csv")
    subprocess.run(
        ["python", "joining_classified.py", "-i", CLASSIFIEDDIR, "-o", output_file],
        check=True,
    )
    logging.info(f"Combined classified data saved to '{output_file}'")


def geocode_data():
    logging.info("Geocoding data...")
    input_file = os.path.join(OUTPUTDIR, "combined_classified_data.csv")
    output_file = os.path.join(OUTPUTDIR, "combined_classified_data_geocoded.csv")
    subprocess.run(
        ["python", "geocode.py", "-i", input_file, "-o", output_file], check=True
    )
    logging.info(f"Geocoded data saved to '{output_file}'")


def group_data_by_location():
    logging.info("Grouping data by location...")
    input_file = os.path.join(OUTPUTDIR, "combined_classified_data_geocoded.csv")
    output_dir = OUTPUTDIR
    subprocess.run(
        ["python", "location_grouping.py", "-i", input_file, "-o", output_dir],
        check=True,
    )
    logging.info(f"Final grouped data saved in '{output_dir}'")


def log_file_stats():
    final_file = os.path.join(OUTPUTDIR, "combined_classified_data_geocoded.csv")
    if not os.path.exists(final_file):
        final_file = os.path.join(OUTPUTDIR, "combined_classified_data.csv")

    if not os.path.exists(final_file):
        logger.error("No final combined data file to log statistics from.")
        return

    df = pd.read_csv(final_file)
    if "DeathDate" in df.columns:
        df["DeathDate"] = pd.to_datetime(df["DeathDate"], errors="coerce")
        df["date"] = df["DeathDate"]
    else:
        logger.warning("No 'DeathDate' column found. Can't compute date ranges.")
        df["date"] = pd.NaT

    if "source_file" not in df.columns:
        logger.warning(
            "No 'source_file' column found. Can't attribute entries to original files."
        )
        return

    # Ensure the logs directory exists
    os.makedirs(os.path.dirname(LOGFILE), exist_ok=True)

    group = df.groupby("source_file")
    with open(LOGFILE, "a") as f:
        for source, subdf in group:
            count = len(subdf)
            min_date = subdf["date"].min()
            max_date = subdf["date"].max()
            if pd.isna(min_date) or pd.isna(max_date):
                date_range_str = "No dates available"
            else:
                date_range_str = (
                    f"{min_date.strftime('%Y/%m')} - {max_date.strftime('%Y/%m')}"
                )
            log_line = (
                f"{source} - {count} new entries in final csv file, {date_range_str}\n"
            )
            f.write(log_line)
            logger.info(log_line.strip())

    group = df.groupby("source_file")
    with open(LOGFILE, "a") as f:
        for source, subdf in group:
            count = len(subdf)
            # Compute date range
            min_date = subdf["date"].min()
            max_date = subdf["date"].max()
            if pd.isna(min_date) or pd.isna(max_date):
                date_range_str = "No dates available"
            else:
                # Format dates as YYYY/MM if required
                date_range_str = (
                    f"{min_date.strftime('%Y/%m')} - {max_date.strftime('%Y/%m')}"
                )
            log_line = (
                f"{source} - {count} new entries in final .csv, {date_range_str}\n"
            )
            f.write(log_line)
            logger.info(log_line.strip())


def join_similar_columns():

    joined_data_loc = os.path.join(OUTPUTDIR, "combined_classified_data.csv")

    dataframe = pd.read_csv(joined_data_loc)

    print("BEFORE")
    print(dataframe.shape)

    column_groups = {
        "CaseNumber": ["CaseNum", "CaseNumber", "Case Number"],
        "ResidenceType": ["ResType", "ResidenceType", "Residence Type"],
        "DeathDate": ["DeathDate", "Date of Death", "Death Date"],
        "DeathTime": ["DeathTime", "Time of Death", "Death Time"],
        "DeathAddress": [
            "DeathAddr",
            "DeathAdress",
            "DeathAddr.1",
            "address.death",
            "DeathAdress.1",
        ],
        "DeathZip": ["DeathZip", "DeathZip.1"],
        "EventPlace": ["EventPlace", "Event Place"],
        "EventAddress": ["EventAddr", "EventAddr.1", "eventaddress", "Event Address"],
        "EventZip": ["EventZip", "EventZip.1"],
        "Mode": ["Mode", "Mode.1"],
        "CauseA": ["CauseA", "Cause A"],
        "CauseB": ["CauseB", "Cause B"],
        "CauseC": ["CauseC", "Cause C"],
        "CauseD": ["CauseD", "Cause D"],
        "CauseOther": ["CauseOther", "Other Cause"],
        "HowInjuryOccurred": ["HowInjuryOccurred", "InjuryDesc"],
        "FirstName": ["First Name"],
        "MiddleName": ["Middle Name"],
        "LastName": ["Last Name"],
        "DateOfBirth": ["Date of Birth"],
        "Text": ["text"],
        "Address": ["address"],  # Need to verify what this refers to
        # Add any other columns as needed
    }

    for new_col, old_cols in column_groups.items():
        # Find which of the old columns exist in the DataFrame
        existing_cols = [col for col in old_cols if col in dataframe.columns]
        if not existing_cols:
            continue  # No columns to merge for this group
        # Combine columns into the new column
        dataframe[new_col] = dataframe[existing_cols].bfill(axis=1).iloc[:, 0]
        # Drop the old columns
        dataframe.drop(
            columns=[col for col in existing_cols if col != new_col], inplace=True
        )

    print("AFTER:")
    print(dataframe.shape)
    dataframe.to_csv(joined_data_loc)


def main():
    rename_files(RAWDIR)
    convert_pdfs()
    classify_csvs()
    join_classified_data()
    join_similar_columns()
    geocode_data()
    group_data_by_location()
    log_file_stats()
    logging.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
