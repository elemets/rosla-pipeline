#!/usr/bin/env python3
"""
Parse the UCLA DME cases PDF into a single, case-level CSV.

PDF structure (learned from inspection):
  - The report is split into many sub-tables, each its own PDF page (form-feed \\x0c
    separated). Every page starts with a header line whose first token is "CaseNum".
  - A page can contain ONE case row or MANY case rows.
  - The columns for a given case are spread across multiple sub-table pages; rows are
    merged back together by CaseNum.
  - Sub-tables come in two shapes:
      * 2-column  -> CaseNum + one wide free-text field (OtherCause, InjuryDesc,
        DeathAddress, EventAddress). These values are NOT left-aligned under their
        header; long ones start far to the left. So column positions are unreliable
        here -> the first token is the CaseNum, everything else is the second column.
      * multi-column -> CaseNum + several short fields. Here values stay close enough
        to their headers that midpoint-boundary bucketing works.
  - Gender/Races leakage is repaired after bucketing: Gender keeps only a recognized
    gender token; extra tokens spill into Races.

Requirements:
  - poppler-utils  (provides `pdftotext` and `pdfinfo`)  -> sudo apt install poppler-utils
  - pandas, typer, tqdm
"""

from __future__ import annotations

import bisect
import multiprocessing as mp
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import typer
from tqdm import tqdm

app = typer.Typer(add_completion=False)

CASE_RE = re.compile(r"^20\d{2}-\d{4,5}$")
PAGE_SEP = "\x0c"

# Some sub-tables name a column slightly differently than our canonical schema.
HEADER_ALIASES = {
    "Race": "Races",
}

# Recognized single-token Gender values (compared lowercased). Anything else sharing
# the Gender cell is treated as race data that leaked across the column boundary.
KNOWN_GENDER_TOKENS = {
    "male", "female", "m", "f", "u",
    "unknown", "undetermined", "non-binary", "nonbinary",
    "null",
}

# ---------- Timestamped logger ----------
_T0 = time.perf_counter()

def log(msg: str) -> None:
    elapsed = time.perf_counter() - _T0
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)

# ---------- Page counting (instant via pdfinfo) ----------
def get_total_pages(pdf_path: str) -> int:
    if not shutil.which("pdfinfo"):
        raise RuntimeError(
            "pdfinfo not found on PATH. Install poppler-utils: sudo apt install poppler-utils"
        )
    out = subprocess.run(
        ["pdfinfo", pdf_path],
        check=True, capture_output=True, text=True, timeout=120,
    ).stdout
    for line in out.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError("Could not parse 'Pages:' line from pdfinfo output.")

# ---------- pdftotext chunk extraction ----------
def extract_pages_text(pdf_path: str, first: int, last: int) -> List[str]:
    """Run `pdftotext -layout -f first -l last`. Returns (last-first+1) page strings."""
    if not shutil.which("pdftotext"):
        raise RuntimeError(
            "pdftotext not found on PATH. Install poppler-utils: sudo apt install poppler-utils"
        )
    result = subprocess.run(
        ["pdftotext", "-layout", "-f", str(first), "-l", str(last), pdf_path, "-"],
        check=True, capture_output=True, text=True,
    )
    pages = result.stdout.split(PAGE_SEP)
    if pages and pages[-1] == "":
        pages.pop()
    expected = last - first + 1
    while len(pages) < expected:
        pages.append("")
    return pages[:expected]

# ---------- Cleaning ----------
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
        # "native american", "american indian", bare "american", "alaskan", "native"
        elif "native" in t or "american indian" in t or "alaskan" in t or t == "american":
            n = "Native American"
        elif "middle" in t:  # "Middle Eastern"
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

# ---------- Text-based page parser ----------
def tokenize_words(line: str) -> List[Tuple[str, int]]:
    """Return [(word, char_start), ...] for whitespace-separated words in `line`."""
    out: List[Tuple[str, int]] = []
    i = 0
    while i < len(line):
        if line[i].isspace():
            i += 1
            continue
        start = i
        while i < len(line) and not line[i].isspace():
            i += 1
        out.append((line[start:i], start))
    return out

def parse_header(line: str) -> Tuple[List[str], List[float]]:
    """
    Return (canonical_column_names, column_boundaries).

    column_boundaries has len(columns) - 1 entries: boundary[i] is the midpoint between
    header start positions of column i and column i+1. A word with start position s
    belongs to column `bisect_right(boundaries, s)`.
    """
    words = tokenize_words(line)
    cols = [HEADER_ALIASES.get(w, w) for w, _ in words]
    starts = [s for _, s in words]
    boundaries = [
        (starts[i - 1] + starts[i]) / 2.0
        for i in range(1, len(starts))
    ]
    return cols, boundaries

def assign_word_to_col(boundaries: List[float], word_start: int, n_cols: int) -> int:
    """Column index for a word, via midpoint boundaries. Clamped to [0, n_cols-1]."""
    idx = bisect.bisect_right(boundaries, word_start)
    return max(0, min(idx, n_cols - 1))

def repair_gender_races(row: Dict[str, str], cols: List[str]) -> None:
    """
    Ensure the Gender cell holds only a recognized gender token. Any extra tokens
    sharing the cell are race words that drifted across the Gender/Races boundary;
    spill them (in order) into the front of the Races cell.

    Mutates `row` in place. No-op for sub-tables that don't have both columns.
    """
    if "Gender" not in cols or "Races" not in cols:
        return
    g = row.get("Gender", "NULL")
    if g in ("NULL", ""):
        return
    tokens = g.split()
    if len(tokens) <= 1:
        return  # already clean
    if tokens[0].lower() not in KNOWN_GENDER_TOKENS:
        return  # first token isn't a gender; leave it alone rather than guess

    row["Gender"] = tokens[0]
    spill = " ".join(tokens[1:]).strip()
    if not spill:
        return
    existing = row.get("Races", "NULL")
    if existing in ("NULL", "", None):
        row["Races"] = spill
    else:
        row["Races"] = f"{spill} {existing}"

def parse_page_text(page_text: str) -> List[Dict[str, str]]:
    """Parse one page's pdftotext output into row dicts."""
    rows: List[Dict[str, str]] = []
    current_cols: Optional[List[str]] = None
    boundaries: Optional[List[float]] = None
    last_row: Optional[Dict[str, str]] = None

    for line in page_text.splitlines():
        if not line.strip():
            continue

        # Header detection: first token must be CaseNum, >=2 columns.
        if "CaseNum" in line:
            cols, bounds = parse_header(line)
            if cols and cols[0] == "CaseNum" and len(cols) >= 2:
                current_cols = cols
                boundaries = bounds
                last_row = None
                continue

        if current_cols is None or boundaries is None:
            continue

        words = tokenize_words(line)
        if not words:
            continue

        first_word, _ = words[0]

        if CASE_RE.match(first_word):
            if len(current_cols) == 2:
                # 2-column sub-table: CaseNum + one wide free-text field whose value
                # is NOT reliably positioned under its header. The first token is the
                # CaseNum (that's how we matched this row); everything else is column 2.
                row = {
                    current_cols[0]: clean_cell(words[0][0]),
                    current_cols[1]: clean_cell(" ".join(w for w, _ in words[1:])),
                }
            else:
                # Multi-column sub-table: values stay near their headers -> midpoint
                # bucketing. Each word goes to the column whose territory it centers in.
                buckets: List[List[str]] = [[] for _ in current_cols]
                for w, w_start in words:
                    buckets[assign_word_to_col(boundaries, w_start, len(current_cols))].append(w)
                row = {
                    col: clean_cell(" ".join(buckets[i]))
                    for i, col in enumerate(current_cols)
                }

            repair_gender_races(row, current_cols)
            rows.append(row)
            last_row = row
        elif last_row is not None and len(current_cols) >= 2:
            # Continuation/wrap line: append to right-most column (matches original behavior).
            extra = clean_cell(line.strip())
            if extra != "NULL":
                last_col = current_cols[-1]
                existing = last_row.get(last_col, "NULL")
                if existing in ("NULL", ""):
                    last_row[last_col] = extra
                else:
                    last_row[last_col] = clean_cell(f"{existing} {extra}")

    return rows

# ---------- Case-level merge ----------
EXPECTED_COLUMNS = [
    "CaseNum", "FirstName", "LastName", "ResType", "DeathDate", "DeathPlace",
    "DeathAddress", "DeathCity", "DeathZip", "EventPlace", "EventAddress",
    "EventCity", "EventZip", "InjuryDesc", "Mode", "DeathCauseA", "DeathCauseB",
    "DeathCauseC", "DeathCauseD", "OtherCause", "Races", "Gender", "Age",
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

def merge_page_into_cases(
    cases: Dict[str, Dict[str, str]],
    page_num: int,
    page_rows: List[Dict[str, str]],
    unknown_cols_seen: Set[str],
) -> Tuple[int, int]:
    """Returns (rows_added, was_parsed)."""
    if not page_rows:
        return 0, 0
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
                unknown_cols_seen.add(k)
                continue
            vv = clean_cell(v)
            if k == "Races" and vv != "NULL":
                vv = normalize_race_value(vv)
            if vv != "NULL":
                rec[k] = vv
            elif rec.get(k, "NULL") in ("", "NULL"):
                rec[k] = "NULL"
    return len(page_rows), 1

# ---------- Parallel worker ----------
def _worker_process_chunk(args: Tuple[str, int, int]) -> List[Tuple[int, List[Dict[str, str]]]]:
    pdf_path, first, last = args
    pages_text = extract_pages_text(pdf_path, first, last)
    return [(first + offset, parse_page_text(pt)) for offset, pt in enumerate(pages_text)]

# ---------- Top-level orchestration ----------
def parse_pdf_to_dataframe(
    pdf_path: str,
    total_pages_override: Optional[int] = None,
    max_pages: Optional[int] = None,
    verbose: bool = True,
    workers: int = 1,
    chunk_size: int = 500,
) -> pd.DataFrame:
    cases: Dict[str, Dict[str, str]] = {}
    parsed_pages = 0
    rows_total = 0
    skipped_pages = 0
    unknown_cols_seen: Set[str] = set()

    if total_pages_override is not None:
        total_pages = total_pages_override
        log(f"Using --total-pages={total_pages}")
    else:
        log("Counting pages via pdfinfo...")
        t = time.perf_counter()
        total_pages = get_total_pages(pdf_path)
        log(f"Page count = {total_pages} ({time.perf_counter() - t:.1f}s)")

    if max_pages is not None and max_pages < total_pages:
        log(f"Capping at --max-pages={max_pages} (of {total_pages})")
        total_pages = max_pages

    chunks: List[Tuple[int, int]] = []
    cur = 1
    while cur <= total_pages:
        chunks.append((cur, min(cur + chunk_size - 1, total_pages)))
        cur += chunk_size

    log(
        f"Mode: {'PARALLEL' if workers > 1 else 'SERIAL'} | workers={workers} "
        f"| chunk_size={chunk_size} | chunks={len(chunks)}"
    )

    if workers <= 1:
        with tqdm(total=total_pages, desc="Pages", unit="page", disable=not verbose) as pbar:
            for first, last in chunks:
                results = _worker_process_chunk((pdf_path, first, last))
                for page_num, page_rows in results:
                    added, was_parsed = merge_page_into_cases(
                        cases, page_num, page_rows, unknown_cols_seen
                    )
                    rows_total += added
                    parsed_pages += was_parsed
                    skipped_pages += (1 - was_parsed)
                    pbar.update(1)
                pbar.set_postfix(cases=len(cases), rows=rows_total, refresh=False)
    else:
        first_result = True
        chunks_done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex, tqdm(
            total=total_pages, desc="Pages", unit="page", disable=not verbose
        ) as pbar:
            t_submit = time.perf_counter()
            futures = [ex.submit(_worker_process_chunk, (pdf_path, f, l)) for f, l in chunks]
            log(f"Submitted {len(futures)} tasks in {time.perf_counter() - t_submit:.1f}s.")

            for fut in as_completed(futures):
                results = fut.result()
                if first_result:
                    tqdm.write(
                        f"[{time.perf_counter() - _T0:7.1f}s] First chunk returned "
                        f"({len(results)} pages). Workers are running."
                    )
                    first_result = False

                for page_num, page_rows in results:
                    added, was_parsed = merge_page_into_cases(
                        cases, page_num, page_rows, unknown_cols_seen
                    )
                    rows_total += added
                    parsed_pages += was_parsed
                    skipped_pages += (1 - was_parsed)
                    pbar.update(1)
                pbar.set_postfix(cases=len(cases), rows=rows_total, refresh=False)

                chunks_done += 1
                if chunks_done % max(1, len(chunks) // 10) == 0:
                    tqdm.write(
                        f"[{time.perf_counter() - _T0:7.1f}s] {chunks_done}/{len(chunks)} chunks done"
                        f" | cases={len(cases)} rows={rows_total}"
                    )

    log("Parsing finished. Building dataframe...")
    df = pd.DataFrame(list(cases.values()))

    for c in EXPECTED_COLUMNS:
        if c not in df.columns:
            df[c] = "NULL"

    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].fillna("NULL").replace({"": "NULL", "nan": "NULL"})

    df = df.rename(
        columns={
            "DeathCauseA": "CauseA",
            "DeathCauseB": "CauseB",
            "DeathCauseC": "CauseC",
            "DeathCauseD": "CauseD",
        }
    )

    final_cols = [
        "CaseNum", "FirstName", "LastName", "ResType", "DeathDate", "DeathPlace",
        "DeathAddress", "DeathCity", "DeathZip", "EventPlace", "EventAddress",
        "EventCity", "EventZip", "InjuryDesc", "Mode", "CauseA", "CauseB",
        "CauseC", "CauseD", "OtherCause", "Races", "Gender", "Age", "SourcePages",
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
        if unknown_cols_seen:
            print(f"  Unknown columns seen (dropped): {sorted(unknown_cols_seen)}")
            print(f"    -> To capture these, add them to EXPECTED_COLUMNS at the top of the script.")

    return df

@app.command()
def main(
    pdf_path: str = typer.Argument(
        "/home/afunnell/Code/Rapid_overdose_clean/pipeline/pipeline_steps/input_files/raw/rawnewdmex/PRA Ruby Romero UCLA All DME Cases 2016 to Present 04152026.pdf",
        help="Path to the input PDF",
    ),
    output_csv: str = typer.Argument("pralatest04152026.csv", help="Path to write the output CSV"),
    total_pages: Optional[int] = typer.Option(None, help="Skip pdfinfo and use this page count."),
    max_pages: Optional[int] = typer.Option(None, help="Only parse the first N pages (for debugging)"),
    workers: int = typer.Option(
        4,
        help="Parallel worker processes. Pass 0 for all CPU cores. 4 is a good default.",
    ),
    chunk_size: int = typer.Option(500, help="Pages per pdftotext invocation."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress output"),
):
    if workers == 0:
        workers = mp.cpu_count()
        log(f"--workers 0 -> using all {workers} CPU cores")

    log(f"Starting. pdf={pdf_path}")
    log(f"Output CSV: {output_csv}")

    df = parse_pdf_to_dataframe(
        pdf_path,
        total_pages_override=total_pages,
        max_pages=max_pages,
        verbose=not quiet,
        workers=workers,
        chunk_size=chunk_size,
    )

    log(f"Writing CSV ({len(df)} rows) to {output_csv}...")
    t_csv = time.perf_counter()
    df.to_csv(output_csv, index=False)
    log(f"CSV written in {time.perf_counter() - t_csv:.1f}s.")

    if not quiet:
        print(f"\nSaved: {output_csv}")
        print("First 5 rows:")
        print(df.head(5).to_string(index=False))

    log("Done.")

if __name__ == "__main__":
    app()