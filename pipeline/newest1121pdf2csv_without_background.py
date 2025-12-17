#!/usr/bin/env python3
"""
Parse LA County DME-style case PDFs into a single CSV.
Hardened for headers that appear with/without spaces.
"""

import re
import csv
from collections import OrderedDict
from typing import List, Dict, Any, Optional
from pathlib import Path

import pdfplumber
import pandas as pd
import typer

app = typer.Typer()

# --- Regex & Configuration ---
CASE_RE = re.compile(r"^\d{4}-\d{5}$")
# Capture group required for pandas .str.extract
ZIP_RE = re.compile(r"(\b\d{5}(?:-\d{4})?\b)")

def norm(s: Optional[str]) -> str:
    """Normalize header-ish strings: lowercase + remove all non-alphanumerics."""
    if s is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

# Identify page types by normalized header keywords
# (no spaces; all-lower; punctuation removed)
PAGE_SIGNATURES_NORM = {
    "main":      ["casenum", "restype", "deathdate"],
    "death_loc": ["deathplace", "deathaddress", "deathcity"],
    "event_loc": ["eventplace", "eventaddress", "eventcity"],
    "injury":    ["injurydesc", "mode"],
    "cause_a":   ["deathcausea", "deathcauseb"],  # C often on same page
    "cause_d":   ["deathcaused", "othercause"],
    "demo":      ["races", "gender", "age"],
}

# Map normalized PDF headers -> clean CSV headers
COLUMN_MAP_NORM = {
    "casenum": "CaseNum",
    "firstname": "FirstName",
    "lastname": "LastName",
    "restype": "ResType",
    # Some PDFs split the header into "D" + "eathDate"
    "eathdate": "DeathDate",
    "deathdate": "DeathDate",

    "deathplace": "DeathPlace",
    "deathaddress": "DeathAddress",
    "deathcity": "DeathCity",
    "deathzip": "DeathZip",

    "eventplace": "EventPlace",
    "eventaddress": "EventAddress",
    "eventcity": "EventCity",
    "eventzip": "EventZip",

    "injurydesc": "InjuryDesc",
    "mode": "Mode",

    "deathcausea": "CauseA",
    "deathcauseb": "CauseB",
    "deathcausec": "CauseC",
    "deathcaused": "CauseD",
    "othercause": "OtherCause",

    "races": "Race",
    "gender": "Gender",
    "age": "Age",
}

TABLE_SETTINGS = {
    # text works well on DME PDFs; you can try "lines" if needed
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 4,
    "join_tolerance": 2,
    "edge_min_length": 3,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
}

def get_page_type(text: str) -> Optional[str]:
    """Determine the page type by looking for normalized header tokens in the text."""
    if not text:
        return None
    t = norm(text)
    for ptype, sigs in PAGE_SIGNATURES_NORM.items():
        # "all-of" is safest; if this proves too strict, relax to ">= 2 matches"
        if all(s in t for s in sigs):
            return ptype
    return None

def clean_cell(cell: Any) -> str:
    """Normalize cell text: remove newlines, strip whitespace, handle NULL."""
    if cell is None:
        return ""
    s = str(cell).replace("\n", " ").strip().strip('"').strip("'")
    return "" if s == "NULL" else s

def merge_fragmented_headers(headers_norm: List[str], row: List[str]) -> (List[str], List[str]):
    """
    PDFs occasionally split headers across two columns (e.g., 'DeathZi' 'p') or
    leave an empty header while shifting the value into the next column.
    Return merged headers and aligned row values.
    """
    merged_headers: List[str] = []
    merged_row: List[str] = []
    i = 0
    while i < len(headers_norm):
        h = headers_norm[i]
        next_h = headers_norm[i + 1] if i + 1 < len(headers_norm) else ""

        # Case 1: two adjacent header fragments make a known key (e.g., deathzi + p)
        if h and next_h and (h + next_h) in COLUMN_MAP_NORM:
            merged_headers.append(h + next_h)
            merged_val = " ".join(v for v in [row[i], row[i + 1]] if v)
            merged_row.append(merged_val)
            i += 2
            continue

        # Case 2: header is known, next header is blank, but the value appears in the next column
        if h in COLUMN_MAP_NORM and not next_h and i + 1 < len(row):
            cur_val = row[i]
            next_val = row[i + 1]
            if not cur_val and next_val:
                merged_headers.append(h)
                merged_row.append(next_val)
                i += 2
                continue

        merged_headers.append(h)
        merged_row.append(row[i] if i < len(row) else "")
        i += 1

    return merged_headers, merged_row

def backfill_zip_from_city(df: pd.DataFrame, city_col: str, zip_col: str) -> None:
    """
    Some PDFs place the ZIP inside the City cell (e.g. 'LOS ANGELES 90012').
    If the explicit ZIP column is blank, pull a 5/9 digit ZIP out of the city cell.
    """
    if city_col not in df.columns:
        return
    if zip_col not in df.columns:
        df[zip_col] = ""

    city_series = df[city_col].fillna("").astype(str)
    zip_series = df[zip_col].fillna("").astype(str)
    found_zips = city_series.str.extract(ZIP_RE)[0]
    needs_fill = zip_series.str.strip().eq("") & found_zips.notna()
    if not needs_fill.any():
        return

    df.loc[needs_fill, zip_col] = found_zips[needs_fill].str.strip()
    df.loc[needs_fill, city_col] = (
        city_series[needs_fill].str.replace(ZIP_RE, "", regex=True).str.rstrip(" ,;")
    )

def find_header_row(table: List[List[Any]], expected_keys_norm: List[str]) -> (int, List[str]):
    """
    Try to locate the header row within a single extracted table using normalized matching.
    Returns (row_index, header_cells) or (-1, []) if not found.
    """
    for i, row in enumerate(table):
        cells = [clean_cell(c) for c in row]
        norms = [norm(c) for c in cells if c is not None]
        # how many of the expected header tokens appear in this row?
        hits = set(expected_keys_norm) & set(norms)
        # require all for short signatures (injury, demo), else at least 2 for longer ones
        need = len(expected_keys_norm)
        min_hits = need if need <= 2 else min(2, need)  # 2 is enough to anchor longer headers
        if len(hits) >= min_hits:
            return i, cells
    return -1, []

def parse_table_to_rows(table: List[List[Any]], header_cells: List[str]) -> List[Dict[str, str]]:
    """Convert a table (list of rows) into dict rows using the provided header row."""
    headers_norm = [norm(clean_cell(h)) for h in header_cells]
    rows: List[Dict[str, str]] = []
    for row in table:
        if not any(c for c in row if c):  # skip all-empty rows
            continue
        cleaned_row = [clean_cell(c) for c in row]
        merged_headers, merged_row = merge_fragmented_headers(headers_norm, cleaned_row)

        # Forward-fill blank headers so multi-column fields (e.g., DeathCauseA split over
        # several columns) keep the same key across adjacent cells.
        filled_headers: List[str] = []
        last = ""
        for h in merged_headers:
            if h:
                last = h
            filled_headers.append(last)

        row_dict: Dict[str, str] = {}
        for i, val in enumerate(merged_row):
            if i >= len(merged_headers):
                continue
            key_norm = filled_headers[i]
            if not key_norm:
                continue
            out_key = COLUMN_MAP_NORM.get(key_norm)
            if out_key:
                # Append text when the same logical column spans multiple extracted cells.
                if out_key in row_dict and val:
                    row_dict[out_key] = f"{row_dict[out_key]} {val}".strip()
                elif val:
                    row_dict[out_key] = val
        if row_dict:
            rows.append(row_dict)
    return rows

def parse_page(page, ptype: str) -> List[Dict[str, str]]:
    """
    Extract rows from a single page by scanning all detected tables and
    picking the one whose header matches the page-type signature.
    """
    tables = page.extract_tables(TABLE_SETTINGS) or []
    if not tables:
        return []

    expected = PAGE_SIGNATURES_NORM[ptype]

    for tbl in tables:
        if not tbl:
            continue
        # locate header row within this table
        header_row_idx, header_cells = find_header_row(tbl, expected)
        if header_row_idx == -1:
            continue

        # data rows are everything after the header; keep the same table slice
        data_slice = tbl[header_row_idx + 1 :]
        if not data_slice:
            continue

        rows = parse_table_to_rows(data_slice, header_cells)
        if rows:
            return rows

    # nothing matched strongly
    return []

def process_pdf(pdf_path: Path, verbose: bool = False) -> pd.DataFrame:
    """Main processing logic."""
    records: "OrderedDict[str, Dict[str, str]]" = OrderedDict()
    current_batch_casenums: List[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        if verbose:
            typer.echo(f"Opened PDF with {len(pdf.pages)} pages.")

        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            ptype = get_page_type(text)

            if not ptype:
                if verbose:
                    typer.echo(f"Skipping Page {i}: Unknown layout.")
                continue

            if verbose:
                typer.echo(f"Processing Page {i} as [{ptype}]")

            extracted_rows = parse_page(page, ptype)

            if not extracted_rows:
                if verbose:
                    typer.echo(f"  Page {i}: No rows extracted for [{ptype}]")
                continue

            if ptype == "main":
                # START OF NEW BATCH: "main" page contains the Case Numbers
                current_batch_casenums = []
                for row in extracted_rows:
                    case_num = row.get("CaseNum")
                    if case_num and CASE_RE.match(case_num):
                        current_batch_casenums.append(case_num)
                        if case_num not in records:
                            records[case_num] = {}
                        # Merge any main-page columns
                        records[case_num].update(row)
                if verbose:
                    typer.echo(f"  New batch with {len(current_batch_casenums)} case(s)")
            else:
                if not current_batch_casenums:
                    if verbose:
                        typer.echo(f"  Warning Page {i}: Data page [{ptype}] but no active CaseNum batch.")
                    continue

                # Align rows by index to the active batch’s CaseNums
                if verbose and len(extracted_rows) != len(current_batch_casenums):
                    typer.echo(f"  Warning Page {i}: Row count mismatch "
                               f"(data={len(extracted_rows)} vs cases={len(current_batch_casenums)})")

                for idx, row_data in enumerate(extracted_rows):
                    if idx < len(current_batch_casenums):
                        case_id = current_batch_casenums[idx]
                        records[case_id].update(row_data)

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(records, orient="index").reset_index(drop=True)

    # If ZIP columns are blank, try to pull them out of the City fields.
    backfill_zip_from_city(df, "DeathCity", "DeathZip")
    backfill_zip_from_city(df, "EventCity", "EventZip")

    # Clean up and reorder columns
    desired_order = [
        "CaseNum", "FirstName", "LastName", "DeathDate", "Age", "Gender", "Race",
        "Mode", "ResType", "DeathPlace", "DeathAddress", "DeathCity", "DeathZip",
        "CauseA", "CauseB", "CauseC", "CauseD", "OtherCause",
        "InjuryDesc", "EventPlace", "EventAddress", "EventCity", "EventZip",
    ]
    final_cols = [c for c in desired_order if c in df.columns]
    extra_cols = [c for c in df.columns if c not in final_cols]
    return df[final_cols + extra_cols]

@app.command()
def main(
    input_pdf: Path = typer.Argument(
        "./pipeline_steps/input_files/raw/LatestRequestPRA11212025.pdf",
        exists=True, file_okay=True, readable=True,
        help="Path to the input PDF file.",
    ),
    output_csv: Path = typer.Option(
        "cases.csv", "--output", "-o", writable=True,
        help="Path to save the output CSV.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
):
    """Convert LA County DME Case PDF to CSV."""
    typer.echo(f"Processing: {input_pdf}")
    try:
        df = process_pdf(input_pdf, verbose)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        typer.echo(typer.style(f"Success! Wrote {len(df)} rows to {output_csv}", fg=typer.colors.GREEN))
    except Exception as e:
        typer.echo(typer.style(f"Error: {e}", fg=typer.colors.RED))
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
