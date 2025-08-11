import os
import glob
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

RAWDIR = "./pipeline_steps/input_files/raw/"
CONVERTEDDIR = "./pipeline_steps/input_files/converted/"
CLASSIFIEDDIR = "./pipeline_steps/input_files/classified/"
OUTPUTDIR = "./pipeline_steps/input_files"
LOGFILE = "./pipeline_steps/logs/pipeline_summary.txt"
MODEL_NAME = "bert_models/bioclinicalbert"


import os
from pathlib import Path

PROCESSED_FILES_LOG = "./pipeline_steps/logs/processed_files.txt"


def fill_from_historical(group, historical_data):
    if len(group) == 1:
        row = group.iloc[0]
        if (
            pd.isna(row["Age"])
            and pd.isna(row["DateofBirth"])
            and not historical_data.empty
        ):
            historical_match = historical_data[
                historical_data["CaseNumber"] == row["CaseNumber"]
            ]
            if not historical_match.empty:
                historical_match = historical_match[
                    historical_match["Age"].notna()
                    | historical_match["DateofBirth"].notna()
                ]
                if not historical_match.empty:
                    latest_record = historical_match.iloc[-1]
                    group.at[row.name, "Age"] = latest_record["Age"]
                    group.at[row.name, "DateofBirth"] = latest_record["DateofBirth"]
        return group

    has_age_data = group["Age"].notna() | group["DateofBirth"].notna()
    if has_age_data.any():
        return group[has_age_data].iloc[[-1]]

    if not historical_data.empty:
        case_number = group["CaseNumber"].iloc[0]
        historical_match = historical_data[
            (historical_data["CaseNumber"] == case_number)
            & (historical_data["Age"].notna() | historical_data["DateofBirth"].notna())
        ]
        if not historical_match.empty:
            latest_historical = historical_match.iloc[-1]
            group.iloc[-1, group.columns.get_loc("Age")] = latest_historical["Age"]
            group.iloc[-1, group.columns.get_loc("DateofBirth")] = latest_historical[
                "DateofBirth"
            ]
            return group.iloc[[-1]]

    return group.iloc[[-1]]


def load_historical_data(geocode_dir):
    """
    Load all previously geocoded files to create a reference dataset.

    Parameters:
        geocode_dir (str): Directory containing geocoded files

    Returns:
        pd.DataFrame: Combined historical data
    """
    historical_files = glob.glob(os.path.join(geocode_dir, "*_geocoded.csv"))
    historical_files = [
        f
        for f in historical_files
        if not f.endswith("combined_classified_data_geocoded.csv")
    ]

    if not historical_files:
        return pd.DataFrame()

    dfs = []
    for file in historical_files:
        try:
            df = pd.read_csv(file)
            if "CaseNumber" in df.columns:
                dfs.append(df)
        except Exception as e:
            logging.warning(f"Could not load {file}: {str(e)}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def load_processed_files():
    if not os.path.exists(PROCESSED_FILES_LOG):
        return set()
    with open(PROCESSED_FILES_LOG, "r") as f:
        return set(line.strip() for line in f if line.strip())


def update_processed_files(new_files):
    with open(PROCESSED_FILES_LOG, "a") as f:
        for file in new_files:
            f.write(file + "\n")


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

    geocode_dir = os.path.join(OUTPUTDIR, "geocoded")
    os.makedirs(geocode_dir, exist_ok=True)  # Ensure the folder exists
    output_file = os.path.join(geocode_dir, "combined_classified_data_geocoded.csv")

    subprocess.run(
        ["python", "geocode.py", "-i", input_file, "-o", output_file], check=True
    )
    logging.info(f"Geocoded data saved to '{output_file}'")


def group_data_by_location():
    logging.info("Grouping data by location...")
    geocode_dir = os.path.join(OUTPUTDIR, "geocoded")
    input_file = os.path.join(geocode_dir, "combined_classified_data_geocoded.csv")
    output_dir = OUTPUTDIR
    subprocess.run(
        ["python", "location_grouping.py", "-i", input_file, "-o", output_dir],
        check=True,
    )
    logging.info(f"Final grouped data saved in '{output_dir}'")


def log_file_stats():
    final_file = os.path.join(
        OUTPUTDIR, "/geocoded/combined_classified_data_geocoded.csv"
    )
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


def join_similar_columns_for_file(input_file, output_file=None):
    """
    Standardizes column names in a single CSV file by joining similar columns.

    Parameters:
      input_file (str): Path to the input CSV file.
      output_file (str): Path to save the output CSV file. If None, the input file is overwritten.
    """
    df = pd.read_csv(input_file)
    print("Before standardization:")
    print(df.columns)

    column_groups = {
        "CaseNumber": ["CaseNum", "CaseNumber", "Case Number", "Case#"],
        "ResidenceType": ["ResType", "ResidenceType", "Residence Type"],
        "DeathDate": [
            "DeathDate",
            "Date of Death",
            "Death Date",
            "DateOfDeath",
            "DateofDeath",
        ],
        "DeathTime": ["DeathTime", "Time of Death", "Death Time", "TimeofDeath"],
        "DeathAddress": [
            "DeathAddress",
            "DeathAddr",
            "DeathAdress",
            "DeathAddr.1",
            "address.death",
            "DeathAdress.1",
            "Death Address",
        ],
        "DeathZip": ["DeathZip", "DeathZip.1", "DeathZipCode", "DeathZi\np", "Zip"],
        "DeathCity": ["DeathCity", "Death City", "DeathCityDesc"],
        "EventPlace": ["EventPlace", "Event Place"],
        "EventAddress": [
            "EventAddress",
            "EventAddr",
            "EventAddr.1",
            "eventaddress",
            "Event Address",
        ],
        "EventZip": [
            "EventZip",
            "EventZip.1",
            "EventZi\np",
            "EventZipCode",
            "Zip.1",
            "Event Zip",
        ],
        "EventCity": ["EventCity", "EventCityDesc"],
        "Mode": ["Mode", "Mode.1"],
        "CauseA": ["CauseA", "Cause A", "DeathCauseA"],
        "CauseB": ["CauseB", "Cause B", "DeathCauseB"],
        "CauseC": ["CauseC", "Cause C", "DeathCauseC"],
        "CauseD": ["CauseD", "Cause D", "DeathCauseD"],
        "CauseOther": ["CauseOther", "Other Cause", "OtherCause"],
        "HowInjuryOccurred": ["HowInjuryOccurred", "InjuryDesc", "HowInjuryOccu\nrred"],
        "FirstName": ["First Name"],
        "MiddleName": ["Middle Name"],
        "LastName": ["Last Name"],
        "DateofBirth": ["Date of Birth", "BirthDate"],
        "Text": ["Text", "text"],
        "Address": ["Address", "address"],
    }

    for new_col, variants in column_groups.items():
        existing_cols = [col for col in variants if col in df.columns]
        if not existing_cols:
            continue

        df[new_col] = df[existing_cols].bfill(axis=1).iloc[:, 0]

        cols_to_drop = [col for col in existing_cols if col != new_col]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)

    print("After standardization:")
    print(df.columns)

    if output_file is None:
        output_file = input_file
    df.to_csv(output_file, index=False)

    print(f"Standardized file saved to: {output_file}")


def get_new_raw_files(raw_dir):
    processed = load_processed_files()
    all_files = [str(p) for p in Path(raw_dir).glob("*") if p.is_file()]
    new_files = [f for f in all_files if f not in processed]
    return new_files


def process_single_file(file_path):
    # Determine the working file for classification.
    if file_path.endswith(".pdf"):
        basename = Path(file_path).stem
        csv_file = os.path.join(CONVERTEDDIR, f"{basename}.csv")
        if not os.path.exists(csv_file):
            logger.info(f"Converting '{file_path}' to '{csv_file}'")
            subprocess.run(
                ["python", "pdf2csv.py", "-i", file_path, "-o", csv_file],
                check=True,
            )
        working_file = csv_file
    else:
        working_file = file_path

    # Classification (if not already classified)
    basename = Path(working_file).stem
    classified_file = os.path.join(CLASSIFIEDDIR, f"{basename}_classified.csv")
    if not os.path.exists(classified_file):
        logger.info(f"Classifying '{working_file}' to '{classified_file}'")
        subprocess.run(
            [
                "python",
                "classify.py",
                "-i",
                working_file,
                "-o",
                classified_file,
                "-m",
                MODEL_NAME,
            ],
            check=True,
        )
    else:
        logger.info(f"'{classified_file}' already exists. Skipping classification.")

    # (Optional) Run join_similar_columns on the file if needed.
    # You can modify your join_similar_columns() function to accept a file path.
    join_similar_columns_for_file(classified_file)

    # Geocode the classified data for this file.
    geocode_dir = os.path.join(OUTPUTDIR, "geocoded")
    os.makedirs(geocode_dir, exist_ok=True)
    geocoded_file = os.path.join(geocode_dir, f"{basename}_geocoded.csv")
    if not os.path.exists(geocoded_file):
        logger.info(f"Geocoding data in '{classified_file}' -> '{geocoded_file}'")
        subprocess.run(
            ["python", "geocode.py", "-i", classified_file, "-o", geocoded_file],
            check=True,
        )
    else:
        logger.info(f"'{geocoded_file}' already exists. Skipping geocoding.")

    append_to_master_geocoded(geocoded_file)


def append_to_master_geocoded(new_geocoded_file):

    geocode_dir = os.path.join(OUTPUTDIR, "geocoded")
    os.makedirs(geocode_dir, exist_ok=True)
    master_file = os.path.join(geocode_dir, "combined_classified_data_geocoded.csv")

    if os.path.exists(master_file):
        master_df = pd.read_csv(master_file)
    else:
        master_df = pd.DataFrame()
    new_df = pd.read_csv(new_geocoded_file)

    combined_df = pd.concat([master_df, new_df], ignore_index=True)

    historical_data = load_historical_data(geocode_dir)

    """
    This ensures that:
    We now check old dataframes for the same data if we are missing data.
    If a CaseNumber has any rows with Age/DOB data, we keep the most recent one with data
    If a CaseNumber has no rows with Age/DOB data, we keep the most recent row
    Single rows are preserved regardless of whether they have Age/DOB data
    """
    if "CaseNumber" in combined_df.columns:
        combined_df = combined_df.groupby("CaseNumber", group_keys=False).apply(
            lambda x: fill_from_historical(x, historical_data)
        )

    # Save the updated master file
    combined_df.to_csv(master_file, index=False)
    print(f"Master geocoded data updated and saved to: {master_file}")


def main():
    rename_files(RAWDIR)

    files_for_processing = get_new_raw_files(RAWDIR)
    for file in files_for_processing:
        process_single_file(file)
    geocode_dir = os.path.join(OUTPUTDIR, "geocoded")
    input_file = os.path.join(geocode_dir, "combined_classified_data_geocoded.csv")
    join_similar_columns_for_file(input_file)
    group_data_by_location()
    update_processed_files(files_for_processing)
    log_file_stats()
    logging.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
