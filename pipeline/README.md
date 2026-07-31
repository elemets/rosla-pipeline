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

## Checking 2025 stability as new reports arrive

The pipeline now writes a stability audit whenever a new raw file is appended to
the master geocoded output. The audit compares the master file immediately
before and after that report is added, filtered to deaths in 2025.

Audit files are written to:

```text
pipeline_steps/audits/
```

For each newly appended report, the audit writes:

- `_summary.csv`: total 2025 cases before and after, net change, added cases,
  removed cases, changed existing cases, and changed cells.
- `_monthly.csv`: 2025 case counts by death month before and after.
- `_added_cases.csv`: the case rows that were newly added.
- `_removed_cases.csv`: case rows that disappeared.
- `_changed_cells.csv`: field-level changes for cases that existed in both
  versions.
- `_added_by_source.csv`: newly added cases grouped by `source_file`.

You can also compare any two existing master files or backups manually:

```bash
python audit_stability.py \
  --old pipeline_steps/input_files/geocoded/combined_classified_data_geocoded.csv.bak-20260604-140155 \
  --new pipeline_steps/input_files/geocoded/combined_classified_data_geocoded.csv \
  --year 2025 \
  --outdir pipeline_steps/audits \
  --prefix manual_2025_check
```

The audit uses `CaseNumber` to decide whether a death is new, removed, or
already present. It parses mixed date formats such as `2025-09-02` and
`9/2/2025`, so 2025 records are counted correctly even when source files format
dates differently.

If you do not have meaningful before/after master backups, reconstruct the
history from the existing per-file geocoded outputs instead:

```bash
python audit_stability.py \
  --reconstruct-dir pipeline_steps/input_files/geocoded \
  --processed-log pipeline_steps/logs/processed_files.txt \
  --year 2025 \
  --outdir pipeline_steps/audits \
  --prefix reconstruct_2025
```

This replays the existing `*_geocoded.csv` files in the order listed in
`processed_files.txt` and writes:

- `_timeline.csv`: one row per added report, with 2025 counts before and after
  that report.
- `_monthly_by_step.csv`: month-level 2025 counts after each report.
- `_added_cases_by_step.csv`: cases that first appear at each report.
- `_removed_cases_by_step.csv`: cases that drop out of 2025 after a later
  report updates the same `CaseNumber`.
- `_changed_cells_by_step.csv`: field-level changes to existing 2025 cases
  after each report.


## NFLIS CITATION

U.S. Drug Enforcement Administration, Diversion Control Division. (2017). Drug Enforcement Administration National Forensic Laboratory Information System: Drug calendar year 2017 data for the National Drug Early Warning System (NDEWS) (Version 1.0). Retrieved May 20, 2026,
