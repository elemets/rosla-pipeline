#!/usr/bin/env python3
"""
Parse LA County DME-style case PDFs into a single CSV.

Key behavior:
- Detect tables by known headers (no fragile page-type signatures).
- Merge rows by CaseNum whenever present; fallback to batch index alignment if not.
- Minimize split words via pdfplumber word-extraction tolerances.
- Repair split-word fragmentation:
    - Address/street style: "Wil ton Pl ace" -> "Wilton Place"
    - Medical/drug ALL CAPS: "FENTAN YL" -> "FENTANYL", "TO XICITY" -> "TOXICITY"
- Always output a stable set of columns (including Injury/Cause fields).
"""

import re
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path

import pdfplumber
import pandas as pd
import typer

app = typer.Typer()

# -----------------------------
# Tunables (start here)
# -----------------------------

# Case numbers are usually YYYY-##### but some files can be YYYY-####.
CASE_RE = re.compile(r"^\d{4}-\d{4,5}$")

# ZIP extraction (for city cells like "LOS ANGELES 90012")
ZIP_RE = re.compile(r"(\b\d{5}(?:-\d{4})?\b)")

# These reduce “word splitting” during table extraction.
# If you ever see columns bleeding together, LOWER TEXT_X_TOL (e.g., 5 -> 4).
TEXT_X_TOL = 6
TEXT_Y_TOL = 3
USE_TEXT_FLOW = True

# Missing value written into the CSV for empty cells
MISSING = ""   # change to "NULL" if that’s preferred

# -----------------------------
# Header normalization & mapping
# -----------------------------

def norm_header(s: Optional[str]) -> str:
    """Normalize header-ish strings: lowercase + remove all non-alphanumerics."""
    if s is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

# Map normalized PDF headers -> clean CSV headers
COLUMN_MAP_NORM: Dict[str, str] = {
    "casenum": "CaseNum",
    "firstname": "FirstName",
    "lastname": "LastName",
    "restype": "ResType",
    "eathdate": "DeathDate",     # handles split "D" + "eathDate"
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

KNOWN_HEADER_KEYS_NORM: Set[str] = set(COLUMN_MAP_NORM.keys())

# -----------------------------
# pdfplumber table settings
# -----------------------------

# NOTE: In pdfplumber 0.11.x, TableSettings.resolve expects text settings
# as keys prefixed with "text_" (not a nested text_settings dict).
TABLE_SETTINGS_PRIMARY = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 4,
    "join_tolerance": 2,
    "edge_min_length": 3,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,

    "text_x_tolerance": TEXT_X_TOL,
    "text_y_tolerance": TEXT_Y_TOL,
    "text_use_text_flow": USE_TEXT_FLOW,
}

# Fallback if primary fails to find tables (rare, but useful)
TABLE_SETTINGS_FALLBACK = {
    **TABLE_SETTINGS_PRIMARY,
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "intersection_tolerance": 5,
}

TABLE_SETTINGS_LIST = [TABLE_SETTINGS_PRIMARY, TABLE_SETTINGS_FALLBACK]

# -----------------------------
# Cell cleaning
# -----------------------------

def clean_cell(cell: Any) -> str:
    """Normalize cell text: remove newlines, strip whitespace, handle NULL."""
    if cell is None:
        return ""
    s = str(cell).replace("\n", " ").strip().strip('"').strip("'")
    return "" if s == "NULL" else s

# -----------------------------
# Split-word repair (combined: medical + street)
# -----------------------------

STOPWORDS = {"of", "and", "the", "in", "at", "to", "from", "on", "by", "vs", "v", "&"}

STREET_SUFFIXES = {
    "st", "street", "ave", "avenue", "blvd", "boulevard", "rd", "road", "dr", "drive",
    "ln", "lane", "ct", "court", "pl", "place", "pkwy", "parkway", "hwy", "highway",
    "fwy", "freeway", "trl", "trail", "cir", "circle", "way", "ter", "terrace",
    "apt", "unit", "ste", "suite",
}

DIRECTIONS = {"n", "s", "e", "w", "ne", "nw", "se", "sw", "nb", "sb", "eb", "wb"}

# Add terms here as you encounter them. Keep them lowercase.
MEDICAL_JOIN_WORDS = {
    # Drugs / tox
    "fentanyl", "methamphetamine", "amphetamine", "cocaine", "ethanol", "heroin",
    "alprazolam", "lorazepam", "hydrocodone", "oxycodone", "trazodone",
    "bromazolam", "fluorofentanyl", "norfentanyl",

    "toxicity", "intoxication", "effects", "effect",

    # Common medical terms that get split
    "cardiovascular", "arteriosclerotic", "atherosclerotic",
    "hypertensive", "hypertension",
    "congestive", "failure",
    "ischemic", "dysfunction", "insufficiency",
    "pulmonary", "thromboembolism", "embolism",
    "obstructive", "obstruction",
    "intracerebral", "subarachnoid", "subdural", "hemorrhage", "aneurysm",
    "sequelae", "idiopathic", "epilepsy", "coccidioidomycosis",
    "cardiopulmonary", "cardiomyopathy", "hypertrophic",

    # Common nouns
    "disease", "injury", "injuries", "trauma", "wound", "wounds",
}
MEDICAL_JOIN_WORDS = {w.lower() for w in MEDICAL_JOIN_WORDS}

ADDRESS_COLS = {"DeathPlace", "DeathAddress", "DeathCity", "EventPlace", "EventAddress", "EventCity"}
MEDICAL_COLS = {"InjuryDesc", "Mode", "CauseA", "CauseB", "CauseC", "CauseD", "OtherCause"}

def _is_stopword(tok: str) -> bool:
    return tok.lower() in STOPWORDS and len(tok) > 1

def _split_affixes(tok: str) -> Tuple[str, str, str]:
    """
    Split token into (prefix_punct, alpha_core, suffix_punct)
    so we can merge 'FENTAN' + 'YL,' -> 'FENTANYL,' preserving comma.
    """
    m = re.match(r'^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$', tok)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return "", tok, ""

def _apply_case(word_lower: str, sample_cores: List[str]) -> str:
    """Apply casing of fragments to the canonical lowercase word."""
    if sample_cores and all(c.isupper() for c in sample_cores):
        return word_lower.upper()
    if sample_cores and all(c.islower() for c in sample_cores):
        return word_lower.lower()
    if sample_cores and sample_cores[0][:1].isupper():
        return word_lower.title()
    return word_lower

def _lexicon_join_tokens(tokens: List[str], lexicon: Set[str], max_join: int = 10) -> List[str]:
    """Join multi-token fragments if their concatenation matches a lexicon word."""
    out: List[str] = []
    i = 0
    while i < len(tokens):
        pref0, core0, suf0 = _split_affixes(tokens[i])

        if core0.isalpha():
            merged = None
            best_L = 0

            for L in range(min(max_join, len(tokens) - i), 1, -1):
                prefs: List[str] = []
                cores: List[str] = []
                sufs: List[str] = []
                ok = True

                for j in range(i, i + L):
                    p, c, s = _split_affixes(tokens[j])
                    if not c.isalpha():
                        ok = False
                        break
                    prefs.append(p)
                    cores.append(c)
                    sufs.append(s)

                if not ok:
                    continue

                cand_lower = "".join(cores).lower()
                if cand_lower in lexicon:
                    merged = prefs[0] + _apply_case(cand_lower, cores) + sufs[-1]
                    best_L = L
                    break

            if merged:
                out.append(merged)
                i += best_L
                continue

        out.append(tokens[i])
        i += 1

    return out

def _heuristic_join(tokens: List[str]) -> List[str]:
    """
    Heuristic joining good for:
      - Street/address: Wil + ton -> Wilton, Pl + ace -> Place
      - Some ALL-CAPS micro-fragments: W + OUN + DS -> WOUNDS
    """
    out: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if i + 1 < len(tokens):
            nxt = tokens[i + 1]

            if tok.isalpha() and nxt.isalpha():
                tok_l = tok.lower()
                nxt_l = nxt.lower()

                tok_is_dir = tok_l in DIRECTIONS
                tok_is_suffix = tok_l in STREET_SUFFIXES
                nxt_is_suffix = nxt_l in STREET_SUFFIXES

                # Avoid joining "FWY south" -> "FWYsouth"
                avoid_suffix_boundary = tok_is_suffix and tok.isupper() and nxt[0].islower()

                merge_lower = (
                    not avoid_suffix_boundary
                    and nxt[0].islower()
                    and 1 <= len(tok) <= 6
                    and 1 <= len(nxt) <= 8
                    and not _is_stopword(tok)
                    and not _is_stopword(nxt)
                    and not (tok_is_dir and nxt_is_suffix)
                )

                merge_upper = (
                    tok.isupper() and nxt.isupper()
                    and 1 <= len(tok) <= 3 and 1 <= len(nxt) <= 3
                    and (len(tok) == 1 or len(nxt) == 1)
                    and not _is_stopword(tok) and not _is_stopword(nxt)
                    and not (tok_is_dir and nxt_is_suffix)
                    and not (tok_is_suffix and len(nxt) == 1)  # avoid AVE + T -> AVET
                )

                if merge_lower or merge_upper:
                    merged = tok + nxt
                    i += 2

                    # Keep consuming additional fragments
                    while i < len(tokens) and tokens[i].isalpha():
                        nxt2 = tokens[i]
                        merged_l = merged.lower()
                        merged_is_suffix = merged_l in STREET_SUFFIXES
                        avoid_suffix_boundary2 = merged_is_suffix and merged.isupper() and nxt2[0].islower()

                        merge_lower2 = (
                            not avoid_suffix_boundary2
                            and nxt2[0].islower()
                            and len(merged) <= 12
                            and 1 <= len(nxt2) <= 8
                            and not _is_stopword(nxt2)
                        )

                        merge_upper2 = (
                            merge_upper
                            and nxt2.isupper()
                            and 1 <= len(nxt2) <= 3
                            and not _is_stopword(nxt2)
                            and not (merged_is_suffix and len(nxt2) == 1)
                        )

                        if merge_lower2 or merge_upper2:
                            merged += nxt2
                            i += 1
                            continue
                        break

                    out.append(merged)
                    continue

        out.append(tok)
        i += 1

    return out

def normalize_value(s: str, kind: str) -> str:
    """
    kind:
      - "address": street-style joining
      - "medical": lexicon joining + glue fixes + heuristic joining
      - "generic": whitespace/punctuation normalization only
    """
    if s is None:
        return ""

    s = str(s).replace("\u00a0", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return s

    # Normalize spacing around punctuation
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s+([,.;:)\]])", r"\1", s)
    s = re.sub(r"([(\[])\s+", r"\1", s)

    # Join sequences of single letters separated by spaces (>=3): "F W Y" -> "FWY"
    s = re.sub(
        r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b",
        lambda m: m.group(0).replace(" ", ""),
        s,
    )

    # Medical-specific “glue” fixes:
    # - "OFABDOMEN" -> "OF ABDOMEN"
    # - "ANDMETHA" -> "AND METHA"
    # - "YLAND" -> "YL AND" (helps "FENTAN YLAND ...")
    if kind == "medical":
        s = re.sub(r"\b(of|and)(?=[A-Za-z]{2,})", r"\1 ", s, flags=re.IGNORECASE)
        s = re.sub(r"\b([A-Za-z]{1,2})(and)\b", r"\1 \2", s, flags=re.IGNORECASE)
        s = re.sub(r"\b([A-Za-z]{2,})(DISEASE)\b", r"\1 \2", s, flags=re.IGNORECASE)

    tokens = s.split(" ")

    if kind == "medical":
        tokens = _lexicon_join_tokens(tokens, MEDICAL_JOIN_WORDS, max_join=10)
        tokens = _heuristic_join(tokens)
    elif kind == "address":
        tokens = _heuristic_join(tokens)

    s = " ".join(tokens)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_field(field: str, value: str) -> str:
    if field in ADDRESS_COLS:
        return normalize_value(value, kind="address")
    if field in MEDICAL_COLS:
        return normalize_value(value, kind="medical")
    return normalize_value(value, kind="generic")

# -----------------------------
# Header fragmentation helpers
# -----------------------------

def merge_fragmented_headers(headers_norm: List[str], row: List[str]) -> Tuple[List[str], List[str]]:
    """
    PDFs sometimes split headers across columns or leave blanks.
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

        # Case 2: known header, next header blank, value appears in next column
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

def row_looks_like_header(row: List[Any]) -> bool:
    """Skip repeated header rows that appear mid-table."""
    norms = [norm_header(clean_cell(c)) for c in row if c is not None]
    hits = len(set(norms) & KNOWN_HEADER_KEYS_NORM)
    if hits >= 3:
        return True
    if "casenum" in norms and hits >= 2:
        return True
    return False

# -----------------------------
# Table parsing
# -----------------------------

def parse_table_to_rows(table: List[List[Any]], header_cells: List[str]) -> List[Dict[str, str]]:
    """Convert table rows into dict rows using the provided header row."""
    headers_norm = [norm_header(clean_cell(h)) for h in header_cells]
    rows: List[Dict[str, str]] = []

    for row in table:
        if not row or not any(c for c in row if c):
            continue
        if row_looks_like_header(row):
            continue

        cleaned_row = [clean_cell(c) for c in row]
        merged_headers, merged_row = merge_fragmented_headers(headers_norm, cleaned_row)

        # Forward-fill blank headers
        filled_headers: List[str] = []
        last = ""
        for h in merged_headers:
            if h:
                last = h
            filled_headers.append(last)

        row_dict: Dict[str, str] = {}
        for i, val in enumerate(merged_row):
            if i >= len(filled_headers):
                continue

            key_norm = filled_headers[i]
            if not key_norm:
                continue

            out_key = COLUMN_MAP_NORM.get(key_norm)
            if not out_key:
                continue

            if out_key in row_dict and val:
                row_dict[out_key] = f"{row_dict[out_key]} {val}".strip()
            elif val:
                row_dict[out_key] = val

        if row_dict:
            # Per-field normalization (medical vs address)
            for k in list(row_dict.keys()):
                if k == "CaseNum":
                    continue
                row_dict[k] = normalize_field(k, row_dict[k])
            rows.append(row_dict)

    return rows

def find_best_header_row(table: List[List[Any]]) -> Tuple[int, List[str], Set[str], int]:
    """
    Pick the row that looks most like a header by matching known header keys.
    Also considers adjacent-cell concatenations (e.g., "DeathZi" + "p").
    """
    best_idx = -1
    best_score = -1
    best_cells: List[str] = []
    best_hits: Set[str] = set()

    for i, row in enumerate(table[:10]):
        cells = [clean_cell(c) for c in row]
        norms = [norm_header(c) for c in cells if c is not None]

        hits: Set[str] = {n for n in norms if n in KNOWN_HEADER_KEYS_NORM}

        # Include adjacent concatenations (handles split headers)
        for j in range(len(norms) - 1):
            comb = norms[j] + norms[j + 1]
            if comb in KNOWN_HEADER_KEYS_NORM:
                hits.add(comb)

        if not hits:
            continue

        score = len(hits) + (10 if "casenum" in hits else 0)
        if score > best_score:
            best_score = score
            best_idx = i
            best_cells = cells
            best_hits = hits

    return best_idx, best_cells, best_hits, best_score

def extract_rows_from_page(page) -> Tuple[List[Dict[str, str]], bool]:
    """
    Extract rows from a page by:
    - extracting tables (primary + fallback settings)
    - choosing the best header row by known headers
    - parsing following rows

    Returns (rows, is_main_page).
    """
    best_rows: List[Dict[str, str]] = []
    best_is_main = False
    best_score = -1.0

    for settings in TABLE_SETTINGS_LIST:
        try:
            tables = page.extract_tables(settings) or []
        except Exception:
            continue

        for tbl in tables:
            if not tbl:
                continue

            header_idx, header_cells, header_hits, score = find_best_header_row(tbl)
            if header_idx == -1:
                continue

            data_slice = tbl[header_idx + 1 :]
            rows = parse_table_to_rows(data_slice, header_cells)
            if not rows:
                continue

            # Prefer tables with better header score + more rows
            score2 = float(score) + min(len(rows), 50) / 10.0
            if score2 > best_score:
                best_score = score2
                best_rows = rows
                best_is_main = ("firstname" in header_hits) or ("lastname" in header_hits)

        # If primary yielded something, don’t waste time on fallback
        if best_rows and settings is TABLE_SETTINGS_PRIMARY:
            break

    return best_rows, best_is_main

# -----------------------------
# ZIP backfill
# -----------------------------

def backfill_zip_from_city(df: pd.DataFrame, city_col: str, zip_col: str) -> None:
    """
    Some PDFs place the ZIP inside the City cell (e.g. 'LOS ANGELES 90012').
    If the explicit ZIP column is blank, pull a 5/9 digit ZIP out of the city cell.
    """
    if city_col not in df.columns:
        return
    if zip_col not in df.columns:
        df[zip_col] = MISSING

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

# -----------------------------
# PDF processing
# -----------------------------

def process_pdf(pdf_path: Path, verbose: bool = False) -> pd.DataFrame:
    records: "OrderedDict[str, Dict[str, str]]" = OrderedDict()
    current_batch_casenums: List[str] = []

    with pdfplumber.open(pdf_path) as pdf:
        if verbose:
            typer.echo(f"Opened PDF with {len(pdf.pages)} pages.")

        for page_idx, page in enumerate(pdf.pages, start=1):
            extracted_rows, is_main = extract_rows_from_page(page)

            if not extracted_rows:
                if verbose:
                    typer.echo(f"Skipping Page {page_idx}: no parseable table found.")
                continue

            if is_main:
                current_batch_casenums = []
                if verbose:
                    typer.echo(f"Page {page_idx}: MAIN page (new batch)")

            for idx, row in enumerate(extracted_rows):
                case_num = row.get("CaseNum")

                # Prefer merge by CaseNum
                if case_num and CASE_RE.match(case_num):
                    records.setdefault(case_num, {}).update(row)
                    if is_main:
                        current_batch_casenums.append(case_num)
                else:
                    # Fallback to batch-index alignment only if CaseNum missing
                    if current_batch_casenums and idx < len(current_batch_casenums):
                        case_id = current_batch_casenums[idx]
                        records.setdefault(case_id, {}).update(row)

    df = pd.DataFrame.from_dict(records, orient="index").reset_index(drop=True)

    # ZIP backfill
    backfill_zip_from_city(df, "DeathCity", "DeathZip")
    backfill_zip_from_city(df, "EventCity", "EventZip")

    desired_order = [
        "CaseNum", "FirstName", "LastName", "DeathDate", "Age", "Gender", "Race",
        "Mode", "ResType", "DeathPlace", "DeathAddress", "DeathCity", "DeathZip",
        "CauseA", "CauseB", "CauseC", "CauseD", "OtherCause",
        "InjuryDesc", "EventPlace", "EventAddress", "EventCity", "EventZip",
    ]

    # Always emit stable headers
    for col in desired_order:
        if col not in df.columns:
            df[col] = MISSING

    # Fill NaNs consistently
    df = df.fillna(MISSING)

    extra_cols = [c for c in df.columns if c not in desired_order]
    return df[desired_order + extra_cols]

# -----------------------------
# CLI
# -----------------------------

@app.command()
def main(
    input_pdf: Path = typer.Argument(
        "./pipeline_steps/input_files/raw/rawnewdmex/LatestRequestPRA11212025.pdf",
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
        df = process_pdf(input_pdf, verbose=verbose)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        typer.echo(typer.style(f"Success! Wrote {len(df)} rows to {output_csv}", fg=typer.colors.GREEN))
    except Exception as e:
        typer.echo(typer.style(f"Error: {e}", fg=typer.colors.RED))
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()