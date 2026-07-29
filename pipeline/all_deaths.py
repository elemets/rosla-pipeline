#!/usr/bin/env python3
"""
Build the all-deaths classified dataset.

Same collation and classification as the overdose pipeline (pipeline.py), minus
the `Any Drugs != 0` filter that classify.py applies on its CLI path, so every
death is kept rather than only the overdoses.

Order matters and mirrors the pipeline:

  1. Classify each raw file *independently* (cached to CLASSIFIED_ALL_DIR).
     Classifying before merging is what makes the output consistent with the
     pipeline: a case counts as an overdose if any source file's row says so.
     Merging first and classifying once lets a stale duplicate row win the
     merge and silently change the label.
  2. Canonicalise column names across files (COL_MAP).
  3. Collapse duplicate CaseNumbers with coalesce_duplicate_rows, taking the
     per-case max of the substance columns first so a 0 cannot mask a 1.
  4. Parse DeathDate, then clean Race / Gender / Mode.

The per-file classified outputs are cached, so re-running is cheap; pass
--force to re-classify from scratch.

Requirements:
  - pandas, numpy, torch, transformers, typer
"""

from __future__ import annotations

import glob
import os
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import typer

from classify import classify_file
from pipeline import coalesce_duplicate_rows

app = typer.Typer(add_completion=False)

RAW_DIR = "./pipeline_steps/input_files/raw"
CLASSIFIED_ALL_DIR = "./pipeline_steps/input_files/classified_all"
MODEL_NAME = "bert_models/bioclinicalbert"

# Canonical column -> possible source columns, first non-missing wins.
COL_MAP = {
    "CaseNumber": ["CaseNumber", "CaseNum", "Case Number", "Case#"],
    "ResidenceType": ["ResidenceType", "ResType", "Residence Type"],
    "DeathDate": ["DeathDate", "Date of Death", "Death Date", "DateOfDeath", "DateofDeath"],
    "DeathTime": ["DeathTime", "Time of Death", "Death Time", "TimeofDeath"],
    "DeathAddress": ["DeathAddress", "DeathAddr", "DeathAdress", "DeathAddr.1", "address.death", "DeathAdress.1", "Death Address"],
    "DeathZip": ["DeathZip", "DeathZip.1", "DeathZipCode", "DeathZi\np", "Zip"],
    "DeathCity": ["DeathCity", "Death City", "DeathCityDesc"],
    "EventPlace": ["EventPlace", "Event Place"],
    "EventAddress": ["EventAddress", "EventAddr", "EventAddr.1", "eventaddress", "Event Address"],
    "EventZip": ["EventZip", "EventZip.1", "EventZi\np", "EventZipCode", "Zip.1", "Event Zip"],
    "EventCity": ["EventCity", "EventCityDesc"],
    "Mode": ["Mode", "Mode.1"],
    "CauseA": ["CauseA", "Cause A", "DeathCauseA"],
    "CauseB": ["CauseB", "Cause B", "DeathCauseB"],
    "CauseC": ["CauseC", "Cause C", "DeathCauseC"],
    "CauseD": ["CauseD", "Cause D", "DeathCauseD"],
    "CauseOther": ["CauseOther", "Other Cause", "OtherCause"],
    "HowInjuryOccurred": ["HowInjuryOccurred", "InjuryDesc", "HowInjuryOccu\nrred"],
    "FirstName": ["FirstName", "First Name"],
    "MiddleName": ["MiddleName", "Middle Name"],
    "LastName": ["LastName", "Last Name"],
    "DateofBirth": ["DateofBirth", "Date of Birth", "BirthDate"],
    "Text": ["Text", "text"],
    "Address": ["Address", "address"],
    "Race": ["Race", "Races"],
}

SUBSTANCE_COLS = [
    "Methamphetamine",
    "Heroin",
    "Cocaine",
    "Fentanyl",
    "Alcohol",
    "Prescription.opioids",
    "Any Opioids",
    "Benzodiazepines",
    "Others",
    "Any Drugs",
]

VALID_RACES = [
    "WHITE",
    "LATINE",
    "BLACK",
    "ASIAN",
    "MIDDLE EASTERN",
    "AMERICAN INDIAN",
    "PACIFIC ISLANDER",
    "UNKNOWN",
]

GENDER_MAP = {
    "M": "MALE",
    "MALE": "MALE",
    "F": "FEMALE",
    "FEMALE": "FEMALE",
    "NON-BINARY": "NON-BINARY",
}


def _blank_to_nan(s: pd.Series) -> pd.Series:
    """Treat empty/whitespace strings as missing."""
    if s.dtype == "object":
        s = s.replace(r"^\s*$", np.nan, regex=True)
    return s


def coalesce_columns(df: pd.DataFrame, col_map: dict[str, list[str]], drop_sources: bool = False) -> pd.DataFrame:
    """Merge each canonical column's aliases into one column, first non-null wins."""
    df = df.copy()

    for canon, aliases in col_map.items():
        present = [c for c in aliases if c in df.columns]
        if not present:
            continue

        out = pd.Series(np.nan, index=df.index)
        for c in present:
            out = out.combine_first(_blank_to_nan(df[c]))

        df[canon] = out

        if drop_sources:
            # Don't drop the canonical column if it was also an alias name.
            df = df.drop(columns=[c for c in present if c != canon], errors="ignore")

    return df


def classify_raw_files(
    raw_dir: str,
    classified_dir: str,
    model_name: str,
    batch_size: int,
    force: bool = False,
) -> pd.DataFrame:
    """
    Classify every raw csv/xlsx separately and return them concatenated.

    Per-file results are cached in `classified_dir` and reused unless `force`.
    """
    os.makedirs(classified_dir, exist_ok=True)

    all_files = sorted(
        glob.glob(os.path.join(raw_dir, "*.csv")) + glob.glob(os.path.join(raw_dir, "*.xlsx"))
    )
    if not all_files:
        raise typer.BadParameter(f"No .csv or .xlsx files found in {raw_dir}")

    dfs = []
    for file in all_files:
        basename = os.path.splitext(os.path.basename(file))[0]
        out_path = os.path.join(classified_dir, f"{basename}_classified.csv")

        if os.path.exists(out_path) and not force:
            typer.echo(f"  cached   {basename}")
            df = pd.read_csv(out_path, low_memory=False)
        else:
            typer.echo(f"  classify {basename}")
            raw = pd.read_excel(file) if file.endswith(".xlsx") else pd.read_csv(file, low_memory=False)
            df = classify_file(raw, out_path, model_name, batch_size=batch_size)
            df["source_file"] = file
            df.to_csv(out_path, index=False)

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def merge_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per CaseNumber, preserving any positive drug flag."""
    # coalesce_duplicate_rows keeps the first non-null value per column, and 0
    # is non-null, so a 0 on the base row would mask a 1 from another row.
    present = [c for c in SUBSTANCE_COLS if c in df.columns]
    if present:
        substance_max = (
            df.groupby("CaseNumber", dropna=False)[present]
            .max()
            .reindex(df["CaseNumber"].values)
        )
        substance_max.index = df.index
        df[present] = substance_max

    # tie_keep="last" matches pipeline.py and location_grouping.py.
    return coalesce_duplicate_rows(df, "CaseNumber", tie_keep="last")


def parse_death_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse DeathDate, recovering dates embedded in otherwise unparseable text."""
    s = df["DeathDate"]

    # Pass 1: parse as-is; unparseable rows become NaT rather than raising.
    death_dt = pd.to_datetime(s, errors="coerce")

    # Pass 2: only fix the failures by extracting a date/datetime token, e.g.
    # "Facility 5/8/2024" or "2022-12-05 00:00:00\tRESIDENCE ...".
    mask = death_dt.isna() & s.notna()
    token = (
        s.loc[mask]
        .astype("string")
        .str.extract(
            r"("
            r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?"  # 2022-12-05 00:00:00
            r"|"
            r"\d{1,2}/\d{1,2}/\d{2,4}"  # 6/1/2023
            r"|"
            r"\d{1,2}-\d{1,2}-\d{2,4}"  # 6-1-2023
            r")",
            expand=False,
        )
    )
    death_dt.loc[mask] = pd.to_datetime(token, errors="coerce")

    df["DeathDate"] = death_dt
    return df.sort_values("DeathDate")


def clean_and_categorize_race(race):
    """Fold the free-text race values into the canonical categories."""
    if pd.isna(race):
        return np.nan

    race = race.lower().replace(" ", "").replace("\n", "")

    # If multiple races are listed in one row, pick the first real one.
    if "," in race:
        parts = [p for p in race.split(",") if p not in ("unknown/other", "unknown", "null")]
        race = parts[0] if parts else "unknown/other"

    if race in ["americanindian", "nativeamerican"]:
        return "AMERICAN INDIAN"
    elif race in ["armenian", "middleeastern"]:
        return "MIDDLE EASTERN"
    elif race in [
        "asian",
        "cambodian",
        "chinese",
        "filipino",
        "japanese",
        "korean",
        "eastindian",
        "thai",
        "vietnamese",
    ]:
        return "ASIAN"
    elif race in ["black"]:
        return "BLACK"
    elif race in [
        "guamanian",
        "hawaiian",
        "pacificislander",
        "samoan",
        "tongan",
        "nativehawaiian/otherpacificislander",
    ]:
        return "PACIFIC ISLANDER"
    elif race in ["hispanic/latino", "hispanic/latina", "hispanic/latinamerican"]:
        return "LATINE"
    elif race in ["white", "caucasian", "white/caucasian"]:
        return "WHITE"
    elif race in ["unknown", "null", "unknown/other"]:
        return np.nan
    else:
        return race


def final_clean(current_df: pd.DataFrame) -> pd.DataFrame:
    """Standardise Mode, Race and Gender; drop race values outside VALID_RACES."""
    current_df["Mode"] = (
        current_df["Mode"].str.replace("\n", "", regex=False).str.strip().str.upper()
    )
    current_df["Mode"] = current_df["Mode"].replace({"UNDETERMI": "UNDETERMINED"})

    current_df["Race"] = (
        current_df["Race"].str.replace("\n", "", regex=False).str.strip().str.upper()
    )
    current_df.loc[~current_df["Race"].isin(VALID_RACES), "Race"] = np.nan

    current_df["Gender"] = (
        current_df["Gender"].str.replace("\n", "", regex=False).str.strip().str.upper()
    )
    current_df["Gender"] = current_df["Gender"].map(GENDER_MAP)

    return current_df


@app.command()
def main(
    output: Optional[str] = typer.Argument(
        None,
        help="Path to write the all-deaths CSV. Defaults to classified_all_deaths_<MMDDYYYY>_regex.csv",
    ),
    raw_dir: str = typer.Option(RAW_DIR, help="Directory of raw csv/xlsx files"),
    classified_dir: str = typer.Option(
        CLASSIFIED_ALL_DIR, help="Where per-file classified outputs are cached"
    ),
    model: str = typer.Option(MODEL_NAME, help="Model name or path under ../models/"),
    batch_size: int = typer.Option(
        1024, help="Classification batch size. Raise it if the GPUs are otherwise idle."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-classify every raw file, ignoring the cache"
    ),
):
    if output is None:
        output = f"classified_all_deaths_{date.today():%m%d%Y}_regex.csv"

    typer.echo(f"Classifying raw files in {raw_dir}")
    combined_df = classify_raw_files(raw_dir, classified_dir, model, batch_size, force=force)
    typer.echo(f"  {len(combined_df):,} rows across all source files")

    combined_df = coalesce_columns(combined_df, COL_MAP, drop_sources=True)
    combined_df = merge_cases(combined_df)
    typer.echo(f"  {len(combined_df):,} unique cases after merging duplicates")

    combined_df = parse_death_dates(combined_df)
    combined_df["Race"] = combined_df["Race"].apply(clean_and_categorize_race)
    combined_df = final_clean(combined_df)

    combined_df.to_csv(output, index=False)
    typer.echo(f"\nSaved: {output}")
    typer.echo(f"  deaths:    {len(combined_df):,}")
    typer.echo(f"  overdoses: {int(combined_df['Any Drugs'].sum()):,}")


if __name__ == "__main__":
    app()
