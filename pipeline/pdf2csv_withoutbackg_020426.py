#!/usr/bin/env python3
"""
Parse the UCLA DME cases PDF into a single, case-level CSV.

This version is designed for PDFs where each page is a text table that begins with a header
containing "CaseNum" and one or more field names (e.g., "DeathPlace", "DeathAddress", etc.).
It uses word x-coordinates (pdfplumber.extract_words) rather than splitting on whitespace, so
multi-word values like "Driveway of residence" or "North 14 FWY south of Ward" stay intact.

Tested against:
- "PRA Ruby Romero UCLA School of Medicine All DME Cases 6-1-2024 to 5-31-2025 02262026.pdf"
"""

from __future__ import annotations
from tqdm import tqdm
import bisect
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pdfplumber
import pandas as pd
import typer

app = typer.Typer(add_completion=False)

# Case number pattern in the PDF looks like "2024-08800" (allow 4–5 digits after dash just in case)
CASE_RE = re.compile(r"^20\d{2}-\d{4,5}$")

# ---------- Optional: race normalization (kept from your original script) ----------
_RACE_TOKEN = r"(?:White\/C\w*|Hisp\w*(?:\/Lat\w*)?|Black|Asian|Native\s+American|Unknown\/Other|Unknown|Other|NULL)"
_RACE_END_RE = re.compile(rf"(?:{_RACE_TOKEN})(?:\s*,\s*(?:{_RACE_TOKEN}))*\s*$", re.IGNORECASE)

def normalize_race_value(raw: str) -> str:
    if raw is None:
        return "NULL"
    raw = str(raw).strip()
    if not raw:
        return "NULL"

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return "NULL"

    normalized: List[str] = []
    seen = set()

    for p in parts:
        t = p.strip().lower()

        if t in ("null", "none", "n/a", "na", ""):
            n = "NULL"
        elif t.startswith("white/c") or "cauc" in t or t == "w" or "white" in t:
            n = "White/Caucasian"
        elif t.startswith("hisp") or "latino" in t or "latina" in t or t == "h":
            n = "Hispanic/Latino"
        elif "black" in t or "african" in t or t in ("b", "aa"):
            n = "Black"
        elif "asian" in t or "pacific" in t or t == "a":
            n = "Asian"
        elif "native" in t:
            n = "Native American"
        elif "unknown" in t or "other" in t:
            n = "Unknown/Other"
        else:
            n = p  # keep as-is if unexpected

        if n not in seen:
            seen.add(n)
            normalized.append(n)

    if all(x == "NULL" for x in normalized):
        return "NULL"
    return ",".join(normalized)

def clean_cell(v: Optional[str]) -> str:
    if v is None:
        return "NULL"
    s = str(v).strip()
    if not s:
        return "NULL"
    if s.lower() == "nan":
        return "NULL"
    if s.upper() in {"NULL", "N/A", "NA", "NONE"}:
        return "NULL"
    s = re.sub(r"\s+", " ", s).strip()
    return s or "NULL"

# ---------- Word->row helpers ----------
@dataclass
class Line:
    top: float
    words: List[dict]

def cluster_lines(words: List[dict], y_tol: float = 2.0) -> List[Line]:
    """Group pdfplumber 'words' into text lines by their y ('top') position."""
    if not words:
        return []

    words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: List[Line] = []

    for w in words_sorted:
        if not lines or abs(w["top"] - lines[-1].top) > y_tol:
            lines.append(Line(top=float(w["top"]), words=[w]))
        else:
            lines[-1].words.append(w)

    for ln in lines:
        ln.words.sort(key=lambda w: w["x0"])
    return lines

def is_header_line(words: List[dict]) -> bool:
    if not words:
        return False
    texts = [w["text"] for w in words]
    return any(t.lower() == "casenum" for t in texts) and len(texts) >= 2

def header_to_columns(header_words: List[dict]) -> Tuple[List[str], List[float]]:
    cols = [(w["text"].strip(), float(w["x0"])) for w in header_words if w["text"].strip()]
    cols.sort(key=lambda x: x[1])
    return [c[0] for c in cols], [c[1] for c in cols]

def assign_word_to_col(x_starts: List[float], x0: float) -> int:
    idx = bisect.bisect_right(x_starts, x0 + 0.01) - 1
    return max(0, min(idx, len(x_starts) - 1))

def parse_page_tables(page: pdfplumber.page.Page) -> List[Dict[str, str]]:
    """
    Parse a page that may contain 1+ tables. Each table begins with a header line containing "CaseNum".
    Returns list of row dicts.
    """
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    if not words:
        return []

    lines = cluster_lines(words, y_tol=2.0)

    rows: List[Dict[str, str]] = []
    current_cols: Optional[List[str]] = None
    x_starts: Optional[List[float]] = None
    last_row: Optional[Dict[str, str]] = None

    for ln in lines:
        if is_header_line(ln.words):
            current_cols, x_starts = header_to_columns(ln.words)
            last_row = None
            continue

        if not current_cols or not x_starts or not ln.words:
            continue

        first_text = ln.words[0]["text"].strip()

        # New row
        if CASE_RE.match(first_text):
            buckets: Dict[str, List[str]] = {c: [] for c in current_cols}

            for w in ln.words:
                col_idx = assign_word_to_col(x_starts, float(w["x0"]))
                buckets[current_cols[col_idx]].append(w["text"])

            row: Dict[str, str] = {c: clean_cell(" ".join(buckets[c]).strip()) for c in current_cols}
            rows.append(row)
            last_row = row
            continue

        # Continuation line (wrap): append to right-most column
        if last_row is not None and len(current_cols) >= 2:
            extra = clean_cell(" ".join([w["text"] for w in ln.words]).strip())
            if extra != "NULL":
                last_col = current_cols[-1]
                last_row[last_col] = clean_cell(
                    f"{'' if last_row.get(last_col) in ('NULL', '') else last_row[last_col] + ' '}{extra}"
                )

    return rows

# ---------- Case-level merge ----------
EXPECTED_COLUMNS = [
    "CaseNum",
    "FirstName",
    "LastName",
    "ResType",
    "DeathDate",
    "DeathPlace",
    "DeathAddress",
    "DeathCity",
    "DeathZip",
    "EventPlace",
    "EventAddress",
    "EventCity",
    "EventZip",
    "InjuryDesc",
    "Mode",
    "DeathCauseA",
    "DeathCauseB",
    "DeathCauseC",
    "DeathCauseD",
    "OtherCause",
    "Races",
    "Gender",
    "Age",
    "SourcePages",
]

def new_case_record(case_num: str) -> Dict[str, str]:
    rec = {c: "NULL" for c in EXPECTED_COLUMNS}
    rec["CaseNum"] = case_num
    rec["SourcePages"] = "NULL"
    return rec

def add_page_to_sources(current: str, page_num: int) -> str:
    if not current or current == "NULL":
        return str(page_num)
    parts = set(p.strip() for p in current.split(",") if p.strip())
    parts.add(str(page_num))
    return ",".join(sorted(parts, key=lambda x: int(x)))

def parse_pdf_to_dataframe(pdf_path: str, max_pages: Optional[int] = None, verbose: bool = True) -> pd.DataFrame:
    cases: Dict[str, Dict[str, str]] = {}
    parsed_pages = 0
    rows_total = 0
    skipped_pages = 0
    print("parsing PDF")

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        if max_pages is not None:
            total_pages = min(total_pages, max_pages)

        for i in tqdm(range(total_pages), desc="Parsing pages", unit="page", disable=not verbose):            
            page_num = i + 1
            page_rows = parse_page_tables(pdf.pages[i])

            if not page_rows:
                skipped_pages += 1
                continue

            parsed_pages += 1
            rows_total += len(page_rows)

            for r in page_rows:
                case = clean_cell(r.get("CaseNum"))
                if case == "NULL" or not CASE_RE.match(case):
                    continue

                if case not in cases:
                    cases[case] = new_case_record(case)

                rec = cases[case]
                rec["SourcePages"] = add_page_to_sources(rec.get("SourcePages", "NULL"), page_num)

                for k, v in r.items():
                    if k == "CaseNum":
                        continue
                    if k not in rec:
                        continue

                    vv = clean_cell(v)
                    if k == "Races" and vv != "NULL":
                        vv = normalize_race_value(vv)

                    if vv != "NULL":
                        rec[k] = vv
                    else:
                        if rec.get(k, "NULL") in ("", "NULL"):
                            rec[k] = "NULL"

            if verbose and page_num % 100 == 0:
                print(f"Processed {page_num}/{total_pages} pages... cases so far: {len(cases)}")

    df = pd.DataFrame(list(cases.values()))

    # Ensure all expected columns exist
    for c in EXPECTED_COLUMNS:
        if c not in df.columns:
            df[c] = "NULL"

    # Standardize NULLs
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].fillna("NULL").replace({"": "NULL", "nan": "NULL"})

    # Rename to match your desired output
    df = df.rename(
        columns={
            "DeathCauseA": "CauseA",
            "DeathCauseB": "CauseB",
            "DeathCauseC": "CauseC",
            "DeathCauseD": "CauseD",
        }
    )

    final_cols = [
        "CaseNum",
        "FirstName",
        "LastName",
        "ResType",
        "DeathDate",
        "DeathPlace",
        "DeathAddress",
        "DeathCity",
        "DeathZip",
        "EventPlace",
        "EventAddress",
        "EventCity",
        "EventZip",
        "InjuryDesc",
        "Mode",
        "CauseA",
        "CauseB",
        "CauseC",
        "CauseD",
        "OtherCause",
        "Races",
        "Gender",
        "Age",
        "SourcePages",
    ]
    for c in final_cols:
        if c not in df.columns:
            df[c] = "NULL"

    df = df[final_cols].sort_values("CaseNum").reset_index(drop=True)

    if verbose:
        print("\nExtraction summary")
        print(f"  Parsed pages (with rows): {parsed_pages}")
        print(f"  Skipped/empty pages:      {skipped_pages}")
        print(f"  Total rows read:          {rows_total}")
        print(f"  Unique cases:             {len(df)}")

    return df

@app.command()
def main(
    pdf_path: str = typer.Argument(..., help="Path to the input PDF"),
    output_csv: str = typer.Argument(..., help="Path to write the output CSV"),
    max_pages: Optional[int] = typer.Option(None, help="Optional: only parse the first N pages (for debugging)"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress output"),
):
    df = parse_pdf_to_dataframe(pdf_path, max_pages=max_pages, verbose=not quiet)
    df.to_csv(output_csv, index=False)
    if not quiet:
        print(f"\nSaved: {output_csv}")
        print("First 5 rows:")
        print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    app()