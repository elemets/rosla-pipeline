import os
import glob
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

RAWDIR = "./pipeline_steps/input_files/raw/"
CONVERTEDDIR = "./pipeline_steps/input_files/converted/"
CLASSIFIEDDIR = "./pipeline_steps/input_files/classified/"
OUTPUTDIR = "./pipeline_steps/input_files"
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


def main():
    rename_files(RAWDIR)
    convert_pdfs()
    classify_csvs()
    join_classified_data()
    geocode_data()
    group_data_by_location()
    logging.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
