# Updated Pipeline Usage

This is the updated pipeline as of 4 June 2026. It now includes the NFLIS-backed
regex classification pass and the general drug-use term regex capture in addition
to the existing BERT classification and geocoding steps.

The pipeline takes raw PDF, CSV, or XLSX files and produces classified,
geocoded overdose records. The main entry point is `pipeline.py`, which handles
conversion, classification, geocoding, joining, and output generation.

## Setup

Create an environment and install the pipeline requirements:

```bash
pip install -r requirements.txt
```

Run all commands from the `pipeline/` directory.

## PDF to CSV conversion

PDF files need to be converted to CSV before they can be classified and
geocoded. Choose the converter based on the PDF format:

- Use `pdf2csv_with_background.py` when the PDF has a background image.
- Use `pdf2csv_withoutbg_051326.py` for newer-format PDFs that do not have a
  background image.
- Use `pdf2csv_withoutbackg_020426.py` for older-format PDFs that do not have a
  background image.

If you are unsure whether a PDF has a background image, open the PDF and inspect
whether the page content sits on top of a repeated image/background layer. PDFs
with that background layer should use the background-aware converter.

If you run one of these converters manually, place the converted CSV in the raw
input folder so `pipeline.py` can classify and geocode it like any other CSV:

```text
pipeline_steps/input_files/raw/
```

## Running the pipeline

Place any new raw `.pdf`, `.csv`, or `.xlsx` files in:

```text
pipeline_steps/input_files/raw/
```

Then run:

```bash
python pipeline.py
```

The pipeline will:

1. Rename raw files to remove spaces.
2. Convert PDFs to CSVs when needed.
3. Classify records with the BERT model.
4. Apply the NFLIS regex classification and general drug-use term regex capture.
5. Standardize similar columns.
6. Geocode the classified records.
7. Append the results to the master geocoded output.
8. Group the final data by location.

The final combined geocoded file is written under:

```text
pipeline_steps/input_files/geocoded/
```

The sorted output is written under:

```text
pipeline_steps/input_files/
```

## Caching and new files

`pipeline.py` caches work at the file level so the full pipeline does not need
to rerun from scratch every time a new file is added.

Processed raw files are tracked in:

```text
pipeline_steps/logs/processed_files.txt
```

The pipeline also skips intermediate work when the expected output already
exists. For example, it will not reconvert a PDF if its converted CSV already
exists, will not reclassify a file if its `_classified.csv` already exists, and
will not geocode a file if its `_geocoded.csv` already exists.

To process additional data, add the new raw file to
`pipeline_steps/input_files/raw/` and run `python pipeline.py` again. Existing
cached files will be reused, and only new or missing work will be performed.


## NFLIS CITATION

U.S. Drug Enforcement Administration, Diversion Control Division. (2017). Drug Enforcement Administration National Forensic Laboratory Information System: Drug calendar year 2017 data for the National Drug Early Warning System (NDEWS) (Version 1.0). Retrieved May 20, 2026,