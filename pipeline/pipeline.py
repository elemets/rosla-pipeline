import os
import glob
import subprocess
import logging
import sys
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
AUDITDIR = "./pipeline_steps/audits"


import os
from pathlib import Path

PROCESSED_FILES_LOG = "./pipeline_steps/logs/processed_files.txt"


def coalesce_duplicate_rows(df, subset, tie_keep="first"):
    """
    Collapse rows that share the same value(s) in `subset` into a single row.

    The row with the most non-null values is used as the base, and any
    remaining nulls are then filled in using values from the other rows in
    the same group, so the result keeps as much data as possible.

    Parameters:
    - df (pd.DataFrame): The DataFrame to deduplicate.
    - subset (str or list): Column name(s) that identify a duplicate.
    - tie_keep (str): For cells where equally complete rows hold *different*
      non-null values, "first" keeps the earlier row's value and "last"
      keeps the later row's value (mirrors drop_duplicates' keep argument).

    Returns:
    - pd.DataFrame: One row per unique `subset` value, with nulls minimized.
    """
    original_columns = df.columns.tolist()

    # Rank rows by how complete they are (most non-null values first).
    completeness = df.notna().sum(axis=1)
    df_ranked = df.assign(_completeness=completeness)

    # Among equally complete rows a *stable* sort preserves the existing row
    # order, so the first row would win ties. Reverse first when the caller
    # wants the last row to win instead.
    if tie_keep == "last":
        df_ranked = df_ranked[::-1]

    df_sorted = df_ranked.sort_values(
        "_completeness", ascending=False, kind="stable"
    ).drop(columns="_completeness")

    # groupby().first() takes the first *non-null* value in each column, so
    # the base (most complete) row's values win and any gaps are filled from
    # the remaining rows in the group.
    combined = df_sorted.groupby(
        subset, as_index=False, sort=False, dropna=False
    ).first()

    # Restore the original column order (groupby moves the key columns first).
    return combined[original_columns]


def fill_from_historical(df, historical_data):
    """
    Fill remaining null values in `df` using previously geocoded records.

    `df` is expected to already be one row per CaseNumber (e.g. the output of
    coalesce_duplicate_rows). For every row that still has nulls, values are
    pulled from historical records sharing the same CaseNumber. Existing
    (non-null) values in `df` are never overwritten.

    Only columns present in both `df` and the historical data are touched, so
    no unexpected columns are introduced into the pipeline.

    Parameters:
        df (pd.DataFrame): Current data, expected to be one row per CaseNumber.
        historical_data (pd.DataFrame): Combined prior geocoded records.

    Returns:
        pd.DataFrame: `df` with nulls filled wherever historical data allowed.
    """
    if historical_data is None or historical_data.empty:
        return df
    if "CaseNumber" not in df.columns or "CaseNumber" not in historical_data.columns:
        return df

    df = df.copy()

    # Collapse historical records down to one best row per CaseNumber so each
    # current row has a single, most-complete historical record to draw from.
    hist = coalesce_duplicate_rows(
        historical_data, subset="CaseNumber", tie_keep="last"
    ).set_index("CaseNumber")

    # Line up one historical row per current row, matched on CaseNumber.
    # Rows with no historical match come back as all-null and change nothing.
    aligned = hist.reindex(df["CaseNumber"].values)
    aligned.index = df.index

    # Fill nulls in df from the aligned historical values; df's own non-null
    # values always win (combine_first only fills where df is null).
    shared_cols = [col for col in df.columns if col in aligned.columns]
    df[shared_cols] = df[shared_cols].combine_first(aligned[shared_cols])

    return df


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
    """
    Regenerate the master geocoded file from the per-file geocoded outputs.

    The master is always rebuilt from scratch rather than appended to. Appending
    made it write-only: a case that stopped classifying as an overdose after a
    raw file was re-extracted had no fresh row left to displace it, so the stale
    row survived every subsequent run. Rebuilding means the per-file geocoded
    outputs are the single source of truth and deleting one actually removes its
    cases.
    """

    geocode_dir = os.path.join(OUTPUTDIR, "geocoded")
    os.makedirs(geocode_dir, exist_ok=True)
    master_file = os.path.join(geocode_dir, "combined_classified_data_geocoded.csv")
    master_existed = os.path.exists(master_file)

    master_df = pd.read_csv(master_file, low_memory=False) if master_existed else None

    # Rebuild from every per-file geocoded output. new_geocoded_file is one of
    # them (process_single_file writes it before calling this), so it is picked
    # up by the glob; guard against it being missed if that ever changes.
    parts = sorted(
        f
        for f in glob.glob(os.path.join(geocode_dir, "*_geocoded.csv"))
        if os.path.abspath(f) != os.path.abspath(master_file)
    )
    if os.path.abspath(new_geocoded_file) not in {os.path.abspath(f) for f in parts}:
        parts.append(new_geocoded_file)

    combined_df = (
        pd.concat([pd.read_csv(f, low_memory=False) for f in parts], ignore_index=True)
        if parts
        else pd.DataFrame()
    )

    historical_data = load_historical_data(geocode_dir)

    # Deduplicate by CaseNumber, then backfill remaining gaps from history:
    #   1. coalesce_duplicate_rows collapses each CaseNumber to a single row,
    #      keeping the most complete row and filling its nulls from the other
    #      duplicate rows (the later row wins when values genuinely conflict).
    #      Unlike the old logic this preserves data across *all* columns, not
    #      just Age / DateofBirth.
    #   2. fill_from_historical fills any values still missing using prior
    #      geocoded files for the same CaseNumber.
    if "CaseNumber" in combined_df.columns:
        combined_df = coalesce_duplicate_rows(
            combined_df, subset="CaseNumber", tie_keep="last"
        )
        combined_df = fill_from_historical(combined_df, historical_data)

    # Save the updated master file
    combined_df.to_csv(master_file, index=False)
    print(f"Master geocoded data updated and saved to: {master_file}")

    if master_existed:
        write_stability_audit(master_df, master_file, new_geocoded_file, year=2025)


def write_stability_audit(before_df, master_file, new_geocoded_file, year=2025):
    os.makedirs(AUDITDIR, exist_ok=True)
    basename = Path(new_geocoded_file).stem.replace("_geocoded", "")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = f"{timestamp}_{basename}_stability_{year}"
    before_file = os.path.join(AUDITDIR, f"{prefix}_before_master.csv")

    before_df.to_csv(before_file, index=False)
    audit_script = Path(__file__).with_name("audit_stability.py")
    try:
        subprocess.run(
            [
                sys.executable,
                str(audit_script),
                "--old",
                before_file,
                "--new",
                master_file,
                "--year",
                str(year),
                "--outdir",
                AUDITDIR,
                "--prefix",
                prefix,
            ],
            check=True,
        )
    finally:
        if os.path.exists(before_file):
            os.remove(before_file)


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
