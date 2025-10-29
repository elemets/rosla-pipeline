#!/usr/bin/env python3
"""
Parse LA County DME-style case PDFs into a single CSV with one row per CaseNum.

Approach
--------
- Use pdfplumber to read words + coordinates.
- Detect page "type" by header text (e.g., 'InjuryDesc Mode DeathCauseA').
- On pages that show a left-hand CaseNum list, use the vertical positions of the
  case numbers to define row bands; assign right-side words to the same band.
- Use the header word x-positions to create column slices, then funnel words
  into the right column for each row.
- Merge results across page types into one dict per CaseNum and export to CSV.

Tested against the structure visible in the sample PDF the user provided
(e.g., pages with headers like 'DeathAddress DeathCity DeathZip EventPlace',
'EventAddress EventCity EventZip', 'InjuryDesc Mode DeathCauseA', etc.).

Usage
-----
pip install pdfplumber pandas
python parse_dme_cases.py input.pdf -o output.csv [--pages 1-300] [--verbose]

Notes
-----
- This script is defensive: it tolerates missing headers on a page,
  empty rows, 'NULL' fields, and wrapped long text.
- If you see OCR quirks, consider nudging the tolerances (x/y_tolerance, gaps).
"""

import argparse
import csv
import re
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Any, Optional

import pdfplumber
import pandas as pd

CASE_RE = re.compile(r"^\d{4}-\d{5}$")
DATE_RE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")

# The final union of columns you'd want in the CSV.
CSV_COLUMNS = [
    "CaseNum",
    "FirstName", "LastName",
    "ResType", "DeathDate", "DeathPlace",
    "DeathAddress", "DeathCity", "DeathZip", "EventPlace",
    "EventAddress", "EventCity", "EventZip",
    "InjuryDesc", "Mode", "DeathCauseA",
    "DeathCauseB", "DeathCauseC", "DeathCauseD",
    "OtherCause", "Races", "Gender", "Age",
]

# Page-type signatures -> expected column headers that appear at page top.
PAGE_TYPES = {
    "main":      ["CaseNum", "FirstName", "LastName", "ResType", "DeathDate", "DeathPlace"],
    "deathloc":  ["DeathAddress", "DeathCity", "DeathZip", "EventPlace"],
    "eventaddr": ["EventAddress", "EventCity", "EventZip"],
    "injury":    ["InjuryDesc", "Mode", "DeathCauseA"],
    "causebc":   ["DeathCauseB", "DeathCauseC", "DeathCauseD"],
    "other":     ["OtherCause", "Races", "Gender", "Age"],
}

# Map page-type to which CSV columns are filled from that page.
PAGE_TYPE_TO_FIELDS = {
    "main":      ["CaseNum", "FirstName", "LastName", "ResType", "DeathDate", "DeathPlace"],
    "deathloc":  ["DeathAddress", "DeathCity", "DeathZip", "EventPlace"],
    "eventaddr": ["EventAddress", "EventCity", "EventZip"],
    "injury":    ["InjuryDesc", "Mode", "DeathCauseA"],
    "causebc":   ["DeathCauseB", "DeathCauseC", "DeathCauseD"],
    "other":     ["OtherCause", "Races", "Gender", "Age"],
}

def collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def page_type_from_text(text: str) -> Optional[str]:
    """Decide which page type this is by finding a unique header signature."""
    t = (text or "").replace("\u00A0", " ")  # non-breaking spaces
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return None
    joined = " ".join(lines)
    # Check unique header sequences in likely priority order
    if "InjuryDesc" in joined and "DeathCauseA" in joined:
        return "injury"
    if "DeathAddress" in joined and "EventPlace" in joined:
        return "deathloc"
    if "EventAddress" in joined and "EventZip" in joined:
        return "eventaddr"
    if "DeathCauseB" in joined and "DeathCauseD" in joined:
        return "causebc"
    if "OtherCause" in joined and "Gender" in joined and "Age" in joined:
        return "other"
    if "ResType" in joined and "DeathDate" in joined and "DeathPlace" in joined:
        return "main"
    return None

def word_center(w: Dict[str, Any]) -> Tuple[float, float]:
    return ( (w["x0"] + w["x1"]) / 2.0, (w["top"] + w["bottom"]) / 2.0 )

def get_words(page) -> List[Dict[str, Any]]:
    # Adjust tolerances if needed for your file
    return page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False)

def find_header_positions(words: List[Dict[str, Any]], headers: List[str]) -> Dict[str, float]:
    """
    Return approximate x-center for each header word that appears near the top.
    Headers are matched literally (case sensitive) on single words as they appear
    on the page (e.g., 'DeathCauseA', 'EventAddress', 'Age').
    """
    if not words:
        return {}

    # Define a limit near the top to avoid picking random occurrences later on the page.
    min_top = min(w["top"] for w in words)
    header_band_bottom = min_top + 90  # pixels; tweak if needed

    x_by_label: Dict[str, float] = {}
    header_set = set(headers)

    for w in words:
        if w["top"] <= header_band_bottom:
            txt = w["text"].strip()
            if txt in header_set and txt not in x_by_label:
                xc, _ = word_center(w)
                x_by_label[txt] = xc

    # Sometimes "FirstName LastName" is printed as two separate headers; that's expected.
    return x_by_label

def midpoint(a: float, b: float) -> float:
    return (a + b) / 2.0

def build_column_bounds(x_centers: List[float], page_width: float, left_limit: Optional[float]=None) -> List[Tuple[float, float]]:
    """
    Given x-centers for header columns, compute [x0, x1] bounds for each column.
    Optionally force a left bound if we need to exclude the CaseNum strip.
    """
    xs = sorted(x_centers)
    bounds: List[Tuple[float, float]] = []
    if not xs:
        return bounds
    # left boundary:
    left = left_limit if left_limit is not None else max(0.0, xs[0] - 20.0)
    for i in range(len(xs)):
        right = midpoint(xs[i], xs[i+1]) if (i+1) < len(xs) else page_width
        bounds.append((left, right))
        left = right
    return bounds

def find_left_case_column(words: List[Dict[str, Any]]) -> Tuple[List[Tuple[str, float]], float]:
    """
    Find case numbers on the page and return ([(case_num, y_center), ...], left_column_right_edge).
    The left_column_right_edge is computed from the max x1 of case number tokens in the left strip.
    """
    case_words = [w for w in words if CASE_RE.fullmatch(w["text"].strip())]
    if not case_words:
        return [], 0.0

    # Identify the "left column" cluster (case numbers in the vertical strip).
    # Heuristic: take the lowest x0 values cluster, based on the minimum x0.
    min_x0 = min(w["x0"] for w in case_words)
    # Allow a ~90px wide strip for the case number column (tweak if needed).
    left_strip = [w for w in case_words if w["x0"] <= (min_x0 + 90)]
    if len(left_strip) < len(case_words) // 3:
        # In rare cases, keep all if the cluster is not obvious
        left_strip = case_words

    pairs = []
    for w in left_strip:
        _, yc = word_center(w)
        pairs.append((w["text"].strip(), yc))

    pairs.sort(key=lambda t: t[1])  # sort by y

    left_edge = max(w["x1"] for w in left_strip) if left_strip else 0.0
    return pairs, left_edge

def y_bands_from_centers(yc: List[float], page_height: float) -> List[Tuple[float, float]]:
    """Convert ordered y-centers into vertical bands for row assignment."""
    if not yc:
        return []
    bands = []
    edges = [0.0] + [midpoint(yc[i], yc[i+1]) for i in range(len(yc)-1)] + [page_height]
    for i in range(len(yc)):
        bands.append((edges[i], edges[i+1]))
    return bands

def assign_words_to_rows(words: List[Dict[str, Any]], bands: List[Tuple[float, float]]) -> List[List[Dict[str, Any]]]:
    """Group words into row-buckets using the vertical bands."""
    rows = [[] for _ in bands]
    for w in words:
        _, yc = word_center(w)
        for i, (y0, y1) in enumerate(bands):
            if y0 <= yc <= y1:
                rows[i].append(w)
                break
    # Sort words in each row by x to preserve reading order
    for r in rows:
        r.sort(key=lambda ww: ww["x0"])
    return rows

def split_row_into_columns(row_words: List[Dict[str, Any]], col_bounds: List[Tuple[float, float]]) -> List[str]:
    """Given the row's words and column [x0,x1] windows, join texts for each column."""
    cols = [""] * len(col_bounds)
    for w in row_words:
        xc, _ = word_center(w)
        for i, (x0, x1) in enumerate(col_bounds):
            if x0 <= xc < x1:
                cols[i] = (cols[i] + " " + w["text"]).strip()
                break
    return [collapse_ws(c) for c in cols]

def parse_main_page(page, words, records: Dict[str, Dict[str, str]], verbose=False):
    """
    Parse the 'main' table: CaseNum, FirstName, LastName, ResType, DeathDate, DeathPlace
    This page typically does NOT have a separate left case-number strip; case numbers sit
    in the same row as other columns, so we build row bands from case number y positions directly.
    """
    page_width = float(page.width)
    page_height = float(page.height)

    # Find main headers to get x positions
    headers = PAGE_TYPES["main"]
    header_x = find_header_positions(words, headers)

    # We need at least the non-name columns
    needed = ["CaseNum", "ResType", "DeathDate", "DeathPlace"]
    if not all(h in header_x for h in needed):
        if verbose:
            print("[main] Warning: missing some headers on page", page.page_number, "found:", header_x)
    # Build bounds using whatever headers we do have (ordered by their x)
    # NOTE: We'll include FirstName and LastName if found; otherwise they stay empty.
    col_order = [h for h in headers if h in header_x]
    col_bounds = build_column_bounds([header_x[h] for h in col_order], page_width)

    # Build row bands from the positions of case numbers that appear in the same table.
    case_words = [w for w in words if CASE_RE.fullmatch(w["text"].strip())]
    case_words.sort(key=lambda w: (word_center(w)[1], word_center(w)[0]))
    yc = [word_center(w)[1] for w in case_words]
    bands = y_bands_from_centers(yc, page_height)
    row_words = assign_words_to_rows(words, bands)

    # Map from col_order to CSV columns (same labels here).
    for i, rw in enumerate(row_words):
        cols = split_row_into_columns(rw, col_bounds)
        row_map = dict(zip(col_order, cols))
        case = collapse_ws(row_map.get("CaseNum", ""))

        # Skip header band or blanks
        if not CASE_RE.fullmatch(case):
            continue

        rec = records.setdefault(case, {k: "" for k in CSV_COLUMNS})
        for label in ["FirstName", "LastName", "ResType", "DeathDate", "DeathPlace"]:
            if label in row_map and row_map[label]:
                rec[label] = row_map[label]

def parse_leftstrip_page(ptype: str, page, words, records: Dict[str, Dict[str, str]], verbose=False):
    """
    Generic parser for pages that show a left CaseNum strip + right-side columns.
    """
    page_width = float(page.width)
    page_height = float(page.height)

    # 1) Find left CaseNum strip and row bands
    pairs, left_right_edge = find_left_case_column(words)
    if not pairs:
        if verbose:
            print(f"[{ptype}] No case numbers found on page {page.page_number}")
        return
    case_list = [c for c, _ in pairs]
    yc = [y for _, y in pairs]
    bands = y_bands_from_centers(yc, page_height)

    # 2) Find headers for this page type to build column x-bounds
    headers = PAGE_TYPES[ptype]
    header_x = find_header_positions(words, headers)
    col_labels = [h for h in headers if h in header_x]
    if not col_labels:
        if verbose:
            print(f"[{ptype}] No headers located on page {page.page_number}; skipping.")
        return
    col_bounds = build_column_bounds([header_x[h] for h in col_labels], page_width, left_limit=left_right_edge + 2)

    # 3) Use only right-side words (beyond the case column)
    right_words = [w for w in words if w["x0"] >= (left_right_edge + 2)]

    # 4) Assign right-side words to row bands
    row_words = assign_words_to_rows(right_words, bands)

    # 5) Split rows into the actual columns for this page type and merge into records
    for i, rw in enumerate(row_words):
        cols = split_row_into_columns(rw, col_bounds)
        row_map = dict(zip(col_labels, cols))
        case = case_list[i] if i < len(case_list) else None
        if not case:
            continue

        rec = records.setdefault(case, {k: "" for k in CSV_COLUMNS})
        for label in headers:
            val = row_map.get(label, "")
            if val:
                rec[label] = val

def parse_pdf(infile: str, pages: Optional[str], verbose: bool=False) -> Dict[str, Dict[str, str]]:
    records: Dict[str, Dict[str, str]] = {}
    with pdfplumber.open(infile) as pdf:
        page_indices = range(len(pdf.pages))
        if pages:
            # pages in "1-60,72,90-100" format
            wanted = set()
            for part in pages.split(","):
                part = part.strip()
                if "-" in part:
                    a, b = part.split("-", 1)
                    start = max(1, int(a))
                    end = min(len(pdf.pages), int(b))
                    wanted.update(range(start, end+1))
                else:
                    p = max(1, int(part))
                    if p <= len(pdf.pages):
                        wanted.add(p)
            # convert to 0-based indices
            page_indices = [p-1 for p in sorted(wanted)]

        for i in page_indices:
            page = pdf.pages[i]
            text = page.extract_text() or ""
            ptype = page_type_from_text(text)

            words = get_words(page)

            if verbose:
                print(f"Page {i+1}/{len(pdf.pages)} type={ptype}")

            if ptype == "main":
                parse_main_page(page, words, records, verbose=verbose)
            elif ptype in ("deathloc", "eventaddr", "injury", "causebc", "other"):
                parse_leftstrip_page(ptype, page, words, records, verbose=verbose)
            else:
                # Some pages may be covers or otherwise not tables we need
                if verbose:
                    print(f"Page {i+1}: Unrecognized or irrelevant; skipping.")

    return records

def clean_columns_from_col_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        s = df[col]
        # Build a mask: non-null cells whose trimmed, casefolded value equals the column name
        mask = s.notna() & s.astype(str).str.strip().str.casefold().eq(col.casefold())
        df.loc[mask, col] = ""

        df[col] = df[col].astype(str).str.replace(
            rf"^\s*{re.escape(col)}\b[:\-]?\s*", "", regex=True
        ).where(df[col].notna(), df[col])
    return df


# Canonical labels you want in the output
ALLOWED_RACES = [
    "Black",
    "Hispanic/Latino",
    "White/Caucasian",
    "Asian",
    "Unknown/Other",
]

# Synonym/variant patterns for each canonical label
RACE_REGEX = {
    "Black": re.compile(r"\bblack\b|\bafrican(?:[-\s]*american)?\b", re.I),
    "Hispanic/Latino": re.compile(r"\bhispanic\b|\blatin(?:o|a|x)?\b", re.I),
    "White/Caucasian": re.compile(r"\bwhite\b|\bcaucasian\b", re.I),
    "Asian": re.compile(r"\basian\b|\b(?:aapi|api)\b", re.I),
    "Unknown/Other": re.compile(r"\bunknown\b|\bother\b|\bunidentified\b", re.I),
}

def extract_race_label(val: object) -> str:
    """Return the race label present in the cell, ignoring any leaked tail text.
    If multiple labels appear, return the one that appears earliest in the text.
    If no label appears, leave the (cleaned) original value unchanged.
    """
    if pd.isna(val):
        return val
    s = str(val)

    # strip accidental header token like "Races: ..."
    s = re.sub(r"^\s*Races\b[:\-]?\s*", "", s, flags=re.I)
    s = " ".join(s.split())  # collapse whitespace
    if s.upper() == "NULL":
        return ""  # optional: treat NULL as empty

    candidates = []
    for label, pat in RACE_REGEX.items():
        m = pat.search(s)
        if m:
            candidates.append((m.start(), label))

    if candidates:
        candidates.sort(key=lambda t: t[0])  # earliest occurrence wins
        return candidates[0][1]

    return s

def main():
    ap = argparse.ArgumentParser(description="Convert DME-style PDF to row-per-CaseNum CSV.")
    ap.add_argument("pdf", help="Input PDF path")
    ap.add_argument("-o", "--output", default="cases.csv", help="Output CSV path")
    ap.add_argument("--pages", default=None,
                    help="Optional page selection, e.g. '1-60' or '1-60,120,500-540'")
    ap.add_argument("--verbose", action="store_true", help="Verbose progress")
    args = ap.parse_args()

    records = parse_pdf(args.pdf, args.pages, verbose=args.verbose)

    # Normalize to DataFrame and write CSV
    # Keep a stable column order; fill missing with empty strings.
    rows = []
    for case, data in records.items():
        row = OrderedDict((col, data.get(col, "")) for col in CSV_COLUMNS)
        # Ensure CaseNum present
        if not row["CaseNum"]:
            row["CaseNum"] = case
        rows.append(row)

    # Sort by CaseNum for readability
    rows.sort(key=lambda r: r.get("CaseNum", ""))

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)\
    
    ## cleaning cause columns and renaming 
    df = clean_columns_from_col_names(df)
    df = df.rename(columns={"DeathCauseA": "CauseA", "DeathCauseB": "CauseB", "DeathCauseC": "CauseC", "DeathCauseD": "CauseD"}, errors="ignore")
    df["Races"] = df["Races"].apply(extract_race_label)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows to {args.output}")

if __name__ == "__main__":
    main()
