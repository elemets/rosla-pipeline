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
      * multi-column -> CaseNum + several short fields. Every cell is left-aligned
        under its header, so each word is bucketed into the last column whose left
        edge it starts at or after.
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
from html import unescape
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import typer
from tqdm import tqdm

app = typer.Typer(add_completion=False)

CASE_RE = re.compile(r"^20\d{2}-\d{4,5}$")
# Points a value may start left of its own header before it is treated as
# belonging to the column on its left.
COL_START_TOLERANCE = 2.0

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
WORD_RE = re.compile(
    r'<word xMin="([\d.eE+-]+)" yMin="([\d.eE+-]+)" xMax="([\d.eE+-]+)" yMax="[\d.eE+-]+">(.*?)</word>'
)
PAGE_RE = re.compile(r"<page ")
# Words whose baselines differ by less than this many points sit on one line.
LINE_TOLERANCE = 3.0
# Horizontal gap (points) that separates one column from the next. Words closer
# than this belong to the same cell; the report's columns are always further apart.
COLUMN_GAP = 8.0

# A word and the horizontal span it occupies.
Word = Tuple[str, float, float]
Line = List[Word]

def group_words_into_lines(words: List[Tuple[float, float, float, str]]) -> List[Line]:
    """Group (y, x0, x1, word) tuples into lines, top-to-bottom then left-to-right."""
    lines: List[Line] = []
    current: List[Tuple[float, float, float, str]] = []
    line_y = 0.0

    def flush() -> None:
        lines.append([(w, x0, x1) for _, x0, x1, w in sorted(current, key=lambda t: t[1])])

    for y, x0, x1, w in sorted(words):
        if current and y - line_y > LINE_TOLERANCE:
            flush()
            current = []
        if not current:
            line_y = y
        current.append((y, x0, x1, w))
    if current:
        flush()
    return lines

def extract_pages_lines(pdf_path: str, first: int, last: int) -> List[List[Line]]:
    """
    Run `pdftotext -bbox -f first -l last`. Returns (last-first+1) pages, each a
    list of lines, each line a list of (word, x_start, x_end) in points.

    Word coordinates come straight from the PDF rather than from `-layout`'s
    character grid: `-layout` shifts a column's text left when its neighbour
    overflows, which makes header-relative positions unreliable for exactly the
    wide free-text columns this report is full of.
    """
    if not shutil.which("pdftotext"):
        raise RuntimeError(
            "pdftotext not found on PATH. Install poppler-utils: sudo apt install poppler-utils"
        )
    result = subprocess.run(
        ["pdftotext", "-bbox", "-f", str(first), "-l", str(last), pdf_path, "-"],
        check=True, capture_output=True, text=True,
    )
    pages: List[List[Line]] = []
    for chunk in PAGE_RE.split(result.stdout)[1:]:
        words = [
            (round(float(y), 1), float(x0), float(x1), unescape(w))
            for x0, y, x1, w in WORD_RE.findall(chunk)
            if w
        ]
        pages.append(group_words_into_lines(words))
    expected = last - first + 1
    while len(pages) < expected:
        pages.append([])
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
def merge_spans(words: List[Word]) -> List[Tuple[float, float]]:
    """Union of the horizontal spans the given words occupy, left to right."""
    blocks: List[Tuple[float, float]] = []
    for _, x0, x1 in sorted(words, key=lambda w: w[1]):
        if blocks and x0 <= blocks[-1][1]:
            blocks[-1] = (blocks[-1][0], max(blocks[-1][1], x1))
        else:
            blocks.append((x0, x1))
    return blocks

def widest_gap(blocks: List[Tuple[float, float]], lo: float, hi: float) -> Optional[float]:
    """Midpoint of the widest uncovered stretch of [lo, hi], or None if fully covered."""
    best = (0.0, None)
    cursor = lo
    for x0, x1 in blocks:
        if x1 <= lo:
            continue
        if x0 >= hi:
            break
        if x0 - cursor > best[0]:
            best = (x0 - cursor, (cursor + x0) / 2.0)
        cursor = max(cursor, x1)
    if hi - cursor > best[0]:
        best = (hi - cursor, (cursor + hi) / 2.0)
    return best[1] if best[0] > COLUMN_GAP else None

def column_boundaries(header: Line, body: List[Line]) -> List[float]:
    """
    Return len(header) - 1 x positions separating the sub-table's columns.

    Boundaries are read off the blank corridor that runs between two adjacent
    headers rather than off a fixed offset from the header text: some sub-tables
    are left-aligned and others centre their cells, so no single offset works for
    both. Every column is separated from the next by whitespace in every row, so
    the widest uncovered stretch between two header texts is their shared edge.
    Where a cell overflows across that whole stretch, fall back to the next
    column's own left edge.
    """
    blocks = merge_spans([w for line in [header] + body for w in line])
    boundaries: List[float] = []
    for i in range(1, len(header)):
        gap = widest_gap(blocks, header[i - 1][2], header[i][1])
        boundaries.append(header[i][1] - COL_START_TOLERANCE if gap is None else gap)
    return boundaries

def parse_header(line: Line) -> List[str]:
    """Canonical column names for a header line."""
    return [HEADER_ALIASES.get(w, w) for w, _, _ in line]

def assign_word_to_col(boundaries: List[float], word: Word, n_cols: int) -> int:
    """Column index for a word, by the midpoint of the span it occupies."""
    _, x0, x1 = word
    idx = bisect.bisect_right(boundaries, (x0 + x1) / 2.0)
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

def bucket_line(words: Line, n_cols: int, boundaries: List[float]) -> List[List[str]]:
    """Split one line's words into per-column word lists by x position."""
    buckets: List[List[str]] = [[] for _ in range(n_cols)]
    for word in words:
        buckets[assign_word_to_col(boundaries, word, n_cols)].append(word[0])
    return buckets


def split_sub_tables(page_lines: List[Line]) -> List[Tuple[List[str], List[Line]]]:
    """Split a page into (column_names, body_lines) per sub-table header found."""
    tables: List[Tuple[List[str], List[Line]]] = []
    for line in page_lines:
        if not line:
            continue
        if any(w == "CaseNum" for w, _, _ in line):
            cols = parse_header(line)
            if cols[0] == "CaseNum" and len(cols) >= 2:
                tables.append((cols, [line]))
                continue
        if tables:
            tables[-1][1].append(line)
    # The header line stays at the front of each body so it counts towards the
    # column corridors; strip it once boundaries have been derived.
    return tables


def parse_sub_table(cols: List[str], lines: List[Line]) -> List[Dict[str, str]]:
    """Parse one sub-table (header line first, then its rows) into row dicts."""
    header, body = lines[0], lines[1:]
    boundaries = column_boundaries(header, body)
    rows: List[Dict[str, str]] = []
    last_row: Optional[Dict[str, str]] = None
    # Wrapped-cell lines seen since the last row, per column. A cell too long for
    # one line spills UPWARDS in this report: the case number sits on the final
    # line of the block, so the overflow arrives before the row it belongs to.
    pending: Optional[List[List[str]]] = None

    for words in body:
        if CASE_RE.match(words[0][0]):
            if len(cols) == 2:
                # 2-column sub-table: CaseNum + one wide free-text field whose value
                # is NOT reliably positioned under its header. The first token is the
                # CaseNum (that's how we matched this row); everything else is column 2.
                buckets = [[words[0][0]], [w for w, _, _ in words[1:]]]
            else:
                buckets = bucket_line(words, len(cols), boundaries)
            if pending is not None:
                buckets = [p + b for p, b in zip(pending, buckets)]
                pending = None
            row = {col: clean_cell(" ".join(buckets[i])) for i, col in enumerate(cols)}
            repair_gender_races(row, cols)
            rows.append(row)
            last_row = row
        else:
            # Overflow line with no case number: hold it for the row below.
            spill = (
                [[], [w for w, _, _ in words]]
                if len(cols) == 2
                else bucket_line(words, len(cols), boundaries)
            )
            pending = spill if pending is None else [p + s for p, s in zip(pending, spill)]

    # Overflow left over at the end of a sub-table belongs to its last row.
    if pending is not None and last_row is not None:
        for i, col in enumerate(cols):
            extra = clean_cell(" ".join(pending[i]))
            if extra == "NULL":
                continue
            existing = last_row.get(col, "NULL")
            last_row[col] = extra if existing in ("NULL", "") else clean_cell(f"{existing} {extra}")

    return rows


def parse_page_text(page_lines: List[Line]) -> List[Dict[str, str]]:
    """Parse one page's words-with-positions into row dicts."""
    rows: List[Dict[str, str]] = []
    for cols, lines in split_sub_tables(page_lines):
        rows.extend(parse_sub_table(cols, lines))
    return rows

# ---------- Case-level merge ----------
EXPECTED_COLUMNS = [
    "CaseNum", "FirstName", "LastName", "BirthDate", "ResType", "DeathDate",
    "DeathPlace",
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
    pages = extract_pages_lines(pdf_path, first, last)
    return [(first + offset, parse_page_text(pg)) for offset, pg in enumerate(pages)]

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
        "CaseNum", "FirstName", "LastName", "BirthDate", "ResType", "DeathDate",
        "DeathPlace",
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