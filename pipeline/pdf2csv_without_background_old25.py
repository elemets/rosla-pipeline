#!/usr/bin/env python3
"""
Parse the older-format UCLA DME cases PDF (LatestRequestPRA11212025.pdf) into a
case-level CSV.

PDF structure (learned from inspection):
  - The report is a wide table split into repeating 7-page groups. Each group
    covers the SAME set of case rows, one column-slice per page:
      1: CaseNum FirstName LastName ResType DeathDate
      2: DeathPlace DeathAddress DeathCity DeathZip
      3: EventPlace EventAddress EventCity EventZip
      4: EventCity InjuryDesc Mode
      5: EventCity DeathCauseA DeathCauseB DeathCauseC
      6: EventCity DeathCauseD OtherCause
      7: EventCity Races Gender Age
    Pages 4-7 repeat EventCity as a carry-over key column; it is ignored.
  - CaseNum appears ONLY on page 1 of each group, so rows on the other pages
    must be matched positionally. Line-index matching fails because long cells
    wrap onto extra lines, but the renderer keeps identical row y-positions on
    every page of a group. Rows are therefore aligned by y-band: each word is
    assigned to the last page-1 CaseNum anchor whose top is above it.
  - Values are bucketed into columns by the midpoint between header x-starts.

Requirements:
  - pdfplumber, pandas, typer, tqdm
"""

from __future__ import annotations

import bisect
import multiprocessing as mp
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pdfplumber
import typer
from tqdm import tqdm

app = typer.Typer(add_completion=False)

CASE_RE = re.compile(r"^20\d{2}-\d{4,6}$")
GROUP_SIZE = 7
HEADER_Y_TOL = 2.0
ROW_Y_TOL = 2.0

HEADER_ALIASES = {"Race": "Races"}

KNOWN_GENDER_TOKENS = {
    "male", "female", "m", "f", "u",
    "unknown", "undetermined", "non-binary", "nonbinary",
    "null",
}

_T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - _T0:7.1f}s] {msg}", flush=True)


# ---------- Cleaning (same conventions as pdf2csv_withoutbg_051326) ----------
def clean_cell(v: Optional[str]) -> str:
    if v is None:
        return "NULL"
    s = str(v).strip()
    if not s or s.lower() == "nan" or s.upper() in {"NULL", "N/A", "NA", "NONE"}:
        return "NULL"
    s = re.sub(r"\s+", " ", s).strip()
    return s or "NULL"


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
        elif t.startswith("hisp") or "latin" in t or t == "h":
            n = "Hispanic/Latino"
        elif "black" in t or "african" in t or t in ("b", "aa"):
            n = "Black"
        elif "asian" in t or "pacific" in t or t == "a":
            n = "Asian"
        elif "native" in t or "american indian" in t or "alaskan" in t or t == "american":
            n = "Native American"
        elif "middle" in t:
            n = "Middle Eastern"
        elif "unknown" in t or "other" in t:
            n = "Unknown/Other"
        else:
            n = p
        if n not in seen:
            seen.add(n)
            normalized.append(n)
    if all(x == "NULL" for x in normalized):
        return "NULL"
    return ",".join(normalized)


def repair_gender_races(rec: Dict[str, str]) -> None:
    g = rec.get("Gender", "NULL")
    if g in ("NULL", ""):
        return
    tokens = g.split()
    if len(tokens) <= 1 or tokens[0].lower() not in KNOWN_GENDER_TOKENS:
        return
    rec["Gender"] = tokens[0]
    spill = " ".join(tokens[1:]).strip()
    if spill:
        existing = rec.get("Races", "NULL")
        rec["Races"] = spill if existing in ("NULL", "") else f"{spill} {existing}"


# ---------- Page parsing ----------
def split_header_and_data(words: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Header words share the page's smallest top y; everything below is data."""
    if not words:
        return [], []
    top0 = min(w["top"] for w in words)
    header = [w for w in words if w["top"] <= top0 + HEADER_Y_TOL]
    data = [w for w in words if w["top"] > top0 + HEADER_Y_TOL]
    return header, data


COL_X_TOL = 2.0


def parse_header(header_words: List[dict]) -> Tuple[List[str], List[float]]:
    """
    Return (canonical column names, x-boundaries between columns).

    Values are left-aligned at their header's x-start and can run most of the
    way to the next header, so the boundary for column i is the NEXT header's
    x-start (with tolerance), not the midpoint.
    """
    header_words = sorted(header_words, key=lambda w: w["x0"])
    cols = [HEADER_ALIASES.get(w["text"], w["text"]) for w in header_words]
    starts = [w["x0"] for w in header_words]
    boundaries = [starts[i] - COL_X_TOL for i in range(1, len(starts))]
    return cols, boundaries


def assign_rows(
    data_words: List[dict],
    anchor_tops: List[float],
    cols: List[str],
    boundaries: List[float],
    skip_col0: bool,
) -> List[Dict[str, str]]:
    """
    Bucket each word into (row, column) cells.

    Row: the last anchor whose top is at or above the word (within ROW_Y_TOL),
    so wrapped lines inside a tall row band land in the same row as its start.
    Column: midpoint bucketing on the word's x0.
    """
    cells: Dict[Tuple[int, int], List[dict]] = {}
    for w in data_words:
        row_idx = bisect.bisect_right(anchor_tops, w["top"] + ROW_Y_TOL) - 1
        if row_idx < 0:
            continue
        col_idx = min(bisect.bisect_right(boundaries, w["x0"]), len(cols) - 1)
        if skip_col0 and col_idx == 0:
            continue
        cells.setdefault((row_idx, col_idx), []).append(w)

    rows: List[Dict[str, str]] = [{} for _ in anchor_tops]
    for (row_idx, col_idx), ws in cells.items():
        ws.sort(key=lambda w: (w["top"], w["x0"]))
        rows[row_idx][cols[col_idx]] = clean_cell(" ".join(w["text"] for w in ws))
    return rows


def parse_group(pdf, first_page_idx: int, n_pages: int) -> List[Dict[str, str]]:
    """Parse one 7-page group into complete case records."""
    page1_words = pdf.pages[first_page_idx].extract_words()
    header, data = split_header_and_data(page1_words)
    header_texts = {w["text"] for w in header}
    if "CaseNum" not in header_texts:
        return []

    anchors = sorted(
        ((w["text"], w["top"]) for w in data if CASE_RE.match(w["text"])),
        key=lambda a: a[1],
    )
    if not anchors:
        return []
    anchor_tops = [a[1] for a in anchors]

    records = [{"CaseNum": case} for case, _ in anchors]
    group_pages = [
        str(p + 1) for p in range(first_page_idx, min(first_page_idx + GROUP_SIZE, n_pages))
    ]
    for rec in records:
        rec["SourcePages"] = ",".join(group_pages)

    for offset in range(GROUP_SIZE):
        page_idx = first_page_idx + offset
        if page_idx >= n_pages:
            break
        words = page1_words if offset == 0 else pdf.pages[page_idx].extract_words()
        header, data = split_header_and_data(words)
        if not header:
            continue
        cols, boundaries = parse_header(header)
        skip_col0 = offset >= 3 and cols and cols[0] == "EventCity"
        for rec, row in zip(records, assign_rows(data, anchor_tops, cols, boundaries, skip_col0)):
            for k, v in row.items():
                if k in ("CaseNum", "EventCity") and offset >= 3:
                    continue
                if v != "NULL":
                    rec[k] = v
    return records


# ---------- Parallel worker ----------
def _worker_process_groups(args: Tuple[str, List[int], int]) -> List[Dict[str, str]]:
    pdf_path, group_starts, n_pages = args
    out: List[Dict[str, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for start in group_starts:
            out.extend(parse_group(pdf, start, n_pages))
    return out


# ---------- Orchestration ----------
EXPECTED_COLUMNS = [
    "CaseNum", "FirstName", "LastName", "BirthDate", "ResType", "DeathDate",
    "DeathPlace",
    "DeathAddress", "DeathCity", "DeathZip", "EventPlace", "EventAddress",
    "EventCity", "EventZip", "InjuryDesc", "Mode", "DeathCauseA", "DeathCauseB",
    "DeathCauseC", "DeathCauseD", "OtherCause", "Races", "Gender", "Age",
    "SourcePages",
]


def parse_pdf_to_dataframe(
    pdf_path: str,
    max_groups: Optional[int] = None,
    workers: int = 4,
    verbose: bool = True,
) -> pd.DataFrame:
    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
    log(f"Page count = {n_pages}")
    if n_pages % GROUP_SIZE != 0:
        log(f"WARNING: page count {n_pages} not divisible by {GROUP_SIZE}; last group may be partial")

    group_starts = list(range(0, n_pages, GROUP_SIZE))
    if max_groups is not None:
        group_starts = group_starts[:max_groups]
    log(f"Groups to parse: {len(group_starts)} | workers={workers}")

    all_records: List[Dict[str, str]] = []
    if workers <= 1:
        with pdfplumber.open(pdf_path) as pdf, tqdm(
            total=len(group_starts), desc="Groups", unit="group", disable=not verbose
        ) as pbar:
            for start in group_starts:
                all_records.extend(parse_group(pdf, start, n_pages))
                pbar.update(1)
    else:
        chunks = [group_starts[i::workers] for i in range(workers)]
        with ProcessPoolExecutor(max_workers=workers) as ex, tqdm(
            total=len(group_starts), desc="Groups", unit="group", disable=not verbose
        ) as pbar:
            futures = [
                ex.submit(_worker_process_groups, (pdf_path, chunk, n_pages))
                for chunk in chunks if chunk
            ]
            for fut in as_completed(futures):
                recs = fut.result()
                all_records.extend(recs)
                pbar.update(len({r["SourcePages"] for r in recs}))

    log(f"Parsed {len(all_records)} case rows. Building dataframe...")

    cases: Dict[str, Dict[str, str]] = {}
    for rec in all_records:
        full = {c: "NULL" for c in EXPECTED_COLUMNS}
        full.update({k: v for k, v in rec.items() if k in full})
        if full["Races"] != "NULL":
            full["Races"] = normalize_race_value(full["Races"])
        repair_gender_races(full)
        cases[full["CaseNum"]] = full

    df = pd.DataFrame(list(cases.values()))
    df = df.rename(columns={
        "DeathCauseA": "CauseA", "DeathCauseB": "CauseB",
        "DeathCauseC": "CauseC", "DeathCauseD": "CauseD",
    })
    final_cols = [
        "CaseNum", "FirstName", "LastName", "BirthDate", "ResType", "DeathDate",
        "DeathPlace",
        "DeathAddress", "DeathCity", "DeathZip", "EventPlace", "EventAddress",
        "EventCity", "EventZip", "InjuryDesc", "Mode", "CauseA", "CauseB",
        "CauseC", "CauseD", "OtherCause", "Races", "Gender", "Age", "SourcePages",
    ]
    df = df[final_cols].sort_values("CaseNum").reset_index(drop=True)

    if verbose:
        non_null = (df != "NULL").mean().round(3)
        print("\nNon-null share per column:")
        print(non_null.to_string())
    return df


@app.command()
def main(
    pdf_path: str = typer.Argument(
        "/home/afunnell/Code/Rapid_overdose_clean/pipeline/pipeline_steps/input_files/raw/rawnewdmex/LatestRequestPRA11212025.pdf",
        help="Path to the input PDF",
    ),
    output_csv: str = typer.Argument("PRADMEC1121_reextracted.csv", help="Path to write the output CSV"),
    max_groups: Optional[int] = typer.Option(None, help="Only parse the first N 7-page groups (debugging)"),
    workers: int = typer.Option(4, help="Parallel worker processes. 0 = all CPU cores."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress output"),
):
    if workers == 0:
        workers = mp.cpu_count()
    log(f"Starting. pdf={pdf_path}")
    df = parse_pdf_to_dataframe(pdf_path, max_groups=max_groups, workers=workers, verbose=not quiet)
    log(f"Writing CSV ({len(df)} rows) to {output_csv}...")
    df.to_csv(output_csv, index=False)
    if not quiet:
        print(f"\nSaved: {output_csv}")
        print(df.head(5).to_string(index=False))
    log("Done.")


if __name__ == "__main__":
    app()
