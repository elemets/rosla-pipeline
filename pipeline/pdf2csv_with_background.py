import pdfplumber
import pandas as pd
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional
import typer

app = typer.Typer()



# Token pattern that also matches truncated values like "White/C" or "Hispani"
_RACE_TOKEN = r"(?:White\/C\w*|Hisp\w*(?:\/Lat\w*)?|Black|Asian|Native\s+American|Unknown\/Other|Unknown|Other|NULL)"
_RACE_END_RE = re.compile(rf"(?:{_RACE_TOKEN})(?:\s*,\s*(?:{_RACE_TOKEN}))*\s*$", re.IGNORECASE)

def normalize_race_value(raw: str) -> str:
    """
    Normalize race values, including truncated ones:
      - 'Hispani' -> 'Hispanic/Latino'
      - 'White/C' -> 'White/Caucasian'
      - 'Unknown/Other' stays as 'Unknown/Other'
      - handles multi-values like 'White/Caucasian,Unknown/Other'
    """
    if raw is None:
        return "NULL"
    raw = str(raw).strip()
    if not raw:
        return "NULL"

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return "NULL"

    normalized = []
    seen = set()

    for p in parts:
        t = p.strip().lower()

        if t in ("null", "none", "n/a", "na", ""):
            n = "NULL"
        elif t.startswith("white/c") or "cauc" in t or "white" in t:
            n = "White/Caucasian"
        elif t.startswith("hisp") or "latino" in t or "latina" in t:
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
            n = p  # keep as-is if it's something unexpected

        if n not in seen:
            seen.add(n)
            normalized.append(n)

    if all(x == "NULL" for x in normalized):
        return "NULL"
    return ",".join(normalized)

def split_race_from_end(text: str) -> tuple[str | None, str]:
    """
    Returns (normalized_race_or_None, remainder_without_race).
    Only matches if the race token(s) appear at the very end of the text.
    """
    if text is None:
        return None, ""
    cleaned = re.sub(r"\s+", " ", str(text).strip())
    if not cleaned:
        return None, ""

    m = _RACE_END_RE.search(cleaned)
    if not m:
        return None, cleaned

    race_raw = m.group(0).strip()
    remainder = cleaned[:m.start()].strip()
    return normalize_race_value(race_raw), remainder

def detect_schema_type(page_text: str) -> Optional[int]:
    if not page_text:
        return None

    lines = page_text.strip().split("\n")

    # Use more lines; pdfplumber sometimes pushes header words down
    header_text = " ".join(lines[:15]).lower()

    # Schema 7: OtherCause + Races (DeathCauseD may be present or not depending on PDF)
    if "othercause" in header_text and "races" in header_text:
        return 7

    # Schema 4: Cause B, C, D
    if "deathcauseb" in header_text and "deathcausec" in header_text and "deathcaused" in header_text:
        return 4

    # Schema 2: Injury page (DeathCauseA may or may not be on this same page)
    if "injurydesc" in header_text and "mode" in header_text:
        return 2

    # Schema 6: Demographics page (sometimes races is missing but gender/age present)
    if "gender" in header_text and "age" in header_text and "races" in header_text:
        return 6
    if "gender" in header_text and "age" in header_text:
        return 6

    # Schema 5: OtherCause-only pages
    if "othercause" in header_text:
        return 5

    # Schema 1: Locations
    if "deathcity" in header_text and "eventplace" in header_text:
        return 1

    # Schema 0: Names/date/place
    if "firstname" in header_text and "lastname" in header_text and "deathdate" in header_text:
        return 0

    # Fallback for cause pages
    if "deathcausea" in header_text and "deathcauseb" in header_text:
        return 3

    return None


    return None  # Unidentified
def analyze_pdf_format(pdf_path: str, sample_pages: int = 100) -> Dict:
    """
    Analyze PDF format to understand variations and patterns.
    This helps identify why parsing might fail on certain pages.
    """
    format_analysis = {
        "table_pages": 0,
        "text_pages": 0,
        "empty_pages": 0,
        "case_number_patterns": Counter(),
        "line_patterns": [],
        "problem_pages": [],
    }

    with pdfplumber.open(pdf_path) as pdf:
        pages_to_check = min(sample_pages, len(pdf.pages))

        for i in range(pages_to_check):
            page = pdf.pages[i]

            # Check if page has tables
            tables = page.extract_tables()
            if tables and any(len(t) > 1 for t in tables):
                format_analysis["table_pages"] += 1

            # Check text
            text = page.extract_text()
            if not text or len(text.strip()) < 10:
                format_analysis["empty_pages"] += 1
                continue

            format_analysis["text_pages"] += 1

            # Analyze case number patterns
            case_nums = re.findall(r"(20\d{2}-\d{4,5})", text)
            for num in case_nums:
                pattern = re.sub(r"\d", "X", num)
                format_analysis["case_number_patterns"][pattern] += 1

            # Check for common issues
            lines = text.split("\n")
            if len(lines) < 5:
                format_analysis["problem_pages"].append(
                    {"page": i + 1, "issue": "too few lines", "line_count": len(lines)}
                )

            # Sample line patterns
            if i < 10:  # First 10 pages
                format_analysis["line_patterns"].append(
                    {
                        "page": i + 1,
                        "first_lines": lines[:3] if len(lines) >= 3 else lines,
                        "case_count": len(case_nums),
                    }
                )

    return format_analysis


def simple_extraction_with_fallback(pdf_path: str) -> pd.DataFrame:
    """
    Simple extraction that tries multiple methods and falls back gracefully.
    MODIFIED: This version detects schema from page content, not page number.
    """
    all_records = []
    extraction_stats = {
        "table_success": 0,
        "text_parse_success": 0,
        "failed_pages": [],
        "partial_pages": [],
        "unidentified_schema_pages": [],
    }

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_idx = page_num + 1

            # --- START MODIFICATION ---
            text = page.extract_text()
            if not text or len(text.strip()) < 10:
                extraction_stats["failed_pages"].append(page_idx)
                continue

            schema_type = detect_schema_type(text)

            if schema_type is None:
                extraction_stats["unidentified_schema_pages"].append(page_idx)
                continue
            # --- END MODIFICATION ---

            # Method 1: Try table extraction
            records = []  # IMPORTANT: reset every page
            try:
                tables = page.extract_tables()
                if tables and len(tables[0]) > 1:
                    records = extract_from_table(tables[0], schema_type, page_idx)
            except Exception:
                records = []

            if records:
                all_records.extend(records)
                extraction_stats["table_success"] += 1
                continue
            # Method 2: Try structured text parsing
            try:
                if text:
                    records = extract_from_text_flexible(text, schema_type, page_idx)
                    if records:
                        all_records.extend(records)
                        extraction_stats["text_parse_success"] += 1
                    else:
                        extraction_stats["partial_pages"].append(page_idx)
            except Exception as e:
                extraction_stats["failed_pages"].append(page_idx)
                print(f"Error on page {page_idx} (Schema {schema_type}): {str(e)}")

    # Print statistics
    print(f"\nExtraction Statistics:")
    print(f"  Table extraction success: {extraction_stats['table_success']} pages")
    print(f"  Text parsing success: {extraction_stats['text_parse_success']} pages")
    print(f"  Failed pages: {len(extraction_stats['failed_pages'])}")
    print(f"  Partial extraction: {len(extraction_stats['partial_pages'])}")
    print(f"  Unidentified schema: {len(extraction_stats['unidentified_schema_pages'])}")

    if extraction_stats["unidentified_schema_pages"][:10]:
        print(f"  First 10 unidentified pages: {extraction_stats['unidentified_schema_pages'][:10]}")

    # Convert to DataFrame

    print("SchemaType counts:", Counter(r["SchemaType"] for r in all_records))
    if all_records:
        df = pd.DataFrame(all_records)
        df = merge_by_case_number(df)
        return df

    return pd.DataFrame()

def extract_from_table(table: List[List], schema_type: int, page_num: int) -> List[Dict]:
    """Extract records from a table structure and produce RawData for downstream parsers."""
    if not table or len(table) < 2:
        return []

    records = []
    for row in table[1:]:
        if not row or not any(str(cell).strip() for cell in row):
            continue

        first_cell = str(row[0]).strip()
        if not re.match(r"^20\d{2}-\d{4,5}$", first_cell):
            continue

        # Join remaining cells into a single RawData string so post_process_raw_data() works
        raw_parts = []
        for v in row[1:]:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                raw_parts.append(s)

        raw = re.sub(r"\s+", " ", " ".join(raw_parts)).strip()
        if not raw:
            raw = "NULL"

        records.append(
            {
                "CaseNum": first_cell,
                "SchemaType": schema_type,
                "PageNum": page_num,
                "RawData": raw,
                "ExtractionMethod": "table",
            }
        )

    return records

def extract_from_text_flexible(
    text: str, schema_type: int, page_num: int
) -> List[Dict]:
    """
    Flexible text extraction that adapts to different formats.
    """
    lines = text.strip().split("\n")
    records = []

    # Skip header lines (be flexible about this)
    start_idx = 0
    for i, line in enumerate(lines[:5]):
        if any(
            keyword in line.lower()
            for keyword in ["casenum", "firstname", "deathdate", "injury"]
        ):
            start_idx = i + 1
            break

    # Process lines
    current_case = None
    current_data = []

    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue

        # Look for case number at start of line
        case_patterns = [
            r"^(20\d{2}-\d{5})\s*(.*)",  # 2023-12345 format
            r"^(20\d{2}-\d{4})\s*(.*)",  # 2023-1234 format
            r"^(\d{4}-\d{4,5})\s*(.*)",  # Any 4-digit year
        ]

        case_match = None
        for pattern in case_patterns:
            match = re.match(pattern, line)
            if match:
                case_match = match
                break

        if case_match:
            # Save previous case
            if current_case and current_data:
                record = {
                    "CaseNum": current_case,
                    "SchemaType": schema_type,
                    "PageNum": page_num,
                    "RawData": " ".join(current_data),
                    "ExtractionMethod": "text",
                }
                records.append(record)

            # Start new case
            current_case = case_match.group(1)
            remaining = case_match.group(2).strip()
            current_data = [remaining] if remaining else []
        else:
            # Accumulate data for current case
            if current_case:
                current_data.append(line)

    # Don't forget last case
    if current_case and current_data:
        record = {
            "CaseNum": current_case,
            "SchemaType": schema_type,
            "PageNum": page_num,
            "RawData": " ".join(current_data),
            "ExtractionMethod": "text",
        }
        records.append(record)

    return records


def merge_by_case_number(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    grouped = df.groupby(["CaseNum", "SchemaType"], dropna=False)
    merged_records = []

    for (case_num, schema_type), group in grouped:
        raw_data = " ".join(group.get("RawData", pd.Series(dtype=object)).fillna("").astype(str)).strip()

        pages_series = group.get("PageNum")
        if pages_series is not None:
            pages = [str(int(p)) for p in pages_series.dropna().tolist() if str(p) != "nan"]
            pages_str = ",".join(sorted(set(pages)))
        else:
            pages_str = ""

        merged_records.append(
            {
                "CaseNum": case_num,
                f"Schema{schema_type}_Data": raw_data if raw_data else "NULL",
                f"Schema{schema_type}_Pages": pages_str if pages_str else "NULL",
            }
        )

    merged_df = pd.DataFrame(merged_records)
    final_df = merged_df.groupby("CaseNum").first().reset_index()
    return final_df


def parse_schema7_data(df: pd.DataFrame, data_col: str) -> pd.DataFrame:
    """
    Parse schema 7 data: DeathCauseD + OtherCause + Races.
    Race is reliably at the end of the line, so we peel it off first.

    Remainder heuristic:
      - If remainder starts with NULL => DeathCauseD = NULL, OtherCause = rest
      - Otherwise keep remainder as OtherCause (safer than guessing DeathCauseD)
    """
    for col in ["DeathCauseD", "OtherCause", "Races"]:
        if col not in df.columns:
            df[col] = "NULL"

    for idx, row in df.iterrows():
        data = row.get(data_col)
        if pd.isna(data) or not str(data).strip():
            continue

        cleaned = re.sub(r"\s+", " ", str(data).strip())

        race, remainder = split_race_from_end(cleaned)
        if race is not None and (str(df.at[idx, "Races"]) in ("NULL", "", "nan") or pd.isna(df.at[idx, "Races"])):
            df.at[idx, "Races"] = race

        remainder = remainder.strip()
        if not remainder or remainder.upper() == "NULL":
            # Nothing else usable
            continue

        # If DeathCauseD is NULL explicitly, peel it off and keep what's left as OtherCause
        if re.match(r"^NULL\b", remainder, re.IGNORECASE):
            rest = re.sub(r"^NULL\b", "", remainder, flags=re.IGNORECASE).strip()
            if str(df.at[idx, "DeathCauseD"]) in ("NULL", "", "nan") or pd.isna(df.at[idx, "DeathCauseD"]):
                df.at[idx, "DeathCauseD"] = "NULL"
            if str(df.at[idx, "OtherCause"]) in ("NULL", "", "nan") or pd.isna(df.at[idx, "OtherCause"]):
                df.at[idx, "OtherCause"] = rest if rest else "NULL"
        else:
            # Ambiguous (could be DeathCauseD or OtherCause); keep it as OtherCause to avoid losing info
            if str(df.at[idx, "OtherCause"]) in ("NULL", "", "nan") or pd.isna(df.at[idx, "OtherCause"]):
                df.at[idx, "OtherCause"] = remainder

    return df

def post_process_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process the raw extracted data to parse into proper columns.
    """
    # Schema definitions
    schemas = {
        0: [
            "FirstName",
            "LastName",
            "ResType",
            "DeathDate",
            "DeathPlace",
            "DeathAddress",
        ],
        1: [
            "DeathCity",
            "DeathZip",
            "EventPlace",
            "EventAddress",
            "EventCity",
            "EventZip",
        ],
        2: ["InjuryDesc", "Mode", "DeathCauseA"], # MODIFIED: Added CauseA
        3: ["DeathCauseB"], # MODIFIED: Was CauseA, CauseB
        4: ["DeathCauseC", "DeathCauseD"], # Now handled by schema 4 parser
        5: ["OtherCause"],
        6: ["Races", "Gender", "Age"],
        7: ["DeathCauseD", "OtherCause", "Races"],
    }

    # Process each schema's data
    for schema_type, columns in schemas.items():
        data_col = f"Schema{schema_type}_Data"

        if data_col in df.columns:
            # Initialize columns
            for col in columns:
                if col not in df.columns: # Only init if not already present (like CauseA from schema 2)
                    df[col] = "NULL"

            # Custom parsing for each schema type
            if schema_type == 0:
                df = parse_schema0_data(df, data_col)
            elif schema_type == 1:
                df = parse_schema1_data(df, data_col)
            elif schema_type == 2: # This parser now handles CauseA
                df = parse_schema2_data(df, data_col)
            elif schema_type == 3:
                # This schema is now likely unused by the detector
                # but if it *is* found, run the original parser
                df = parse_schema3_data(df, data_col)
            elif schema_type == 4: # This parser now handles B, C, and D
                df = parse_schema4_data(df, data_col)
            elif schema_type == 5:
                df = parse_schema5_data(df, data_col)
            elif schema_type == 6:
                df = parse_schema6_data(df, data_col)
            elif schema_type == 7:
                df = parse_schema7_data(df, data_col)

    return df


def parse_schema0_data(df: pd.DataFrame, data_col: str) -> pd.DataFrame:
    """Parse schema 0 (page 1) data - Names, ResType, Date, Place."""
    for idx, row in df.iterrows():
        data = row[data_col]
        if pd.isna(data) or not data:
            continue

        # Extract date
        date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", data)
        if date_match:
            df.at[idx, "DeathDate"] = date_match.group(1)

            # Parse before and after date
            before = data[: date_match.start()].strip()
            after = data[date_match.end() :].strip()

            # Parse residence type from before section
            res_types = [
                "Residence",
                "NULL",
                "Trailer",
                "Sober Living",
                "Nursing Home",
                "Shelter",
                "Hotel",
                "Motel",
                "Apartment",
            ]
            for res_type in res_types:
                if res_type.lower() in before.lower():
                    df.at[idx, "ResType"] = res_type
                    # Remove res type to parse names
                    before = re.sub(
                        rf"\b{res_type}\b", "", before, flags=re.IGNORECASE
                    ).strip()
                    break

            # Parse names from remaining before text
            if before and before != "NULL":
                name_parts = before.split()
                if len(name_parts) >= 2:
                    df.at[idx, "FirstName"] = name_parts[0]
                    df.at[idx, "LastName"] = " ".join(name_parts[1:])
                elif len(name_parts) == 1:
                    df.at[idx, "LastName"] = name_parts[0]

            # Parse death place from after section
            if after:
                # Common death places
                death_places = [
                    "Hospital",
                    "Residence",
                    "Street",
                    "Sidewalk",
                    "SIDEWALK",
                    "Freeway",
                    "Church",
                    "School",
                    "Park",
                    "Office",
                    "Restaurant",
                    "Tent",
                    "Hotel",
                    "hotel",
                    "general",
                ]

                words = after.split()
                if words:
                    # Check if first word is a known place
                    place_found = False
                    for place in death_places:
                        if place.lower() == words[0].lower():
                            df.at[idx, "DeathPlace"] = place
                            place_found = True
                            if len(words) > 1:
                                df.at[idx, "DeathAddress"] = " ".join(words[1:])
                            break

                    if not place_found:
                        df.at[idx, "DeathPlace"] = words[0]
                        if len(words) > 1:
                            df.at[idx, "DeathAddress"] = " ".join(words[1:])

    return df


def parse_schema1_data(df: pd.DataFrame, data_col: str) -> pd.DataFrame:
    """Parse schema 1 (locations). Uses DeathCity-derived city set to split EventAddress vs EventCity."""
    street_suffixes = {
        "st","street","ave","avenue","blvd","boulevard","rd","road","dr","drive",
        "ln","lane","ct","court","way","hwy","highway","pkwy","parkway","fw","freeway",
        "pl","place","ter","terrace","cir","circle","trl","trail","pkwy","pkwy."
    }

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip())

    # ---- Pass 1: build a city vocabulary from the DeathCity field in this same PDF ----
    city_set = set()
    for raw in df[data_col].fillna("").astype(str):
        zips = re.findall(r"\b(\d{5})(?:-\d{4})?\b", raw)
        if not zips:
            continue
        no_zips = re.sub(r"\b\d{5}(?:-\d{4})?\b", "|", raw)
        parts = no_zips.split("|")
        if parts and parts[0].strip():
            death_city = norm(parts[0])
            if death_city and death_city.upper() != "NULL":
                city_set.add(death_city)

    # optional: include a few normalized variants
    # (helps match "LOS ANGELES" vs "Los Angeles")
    city_set_ci = {c.lower(): c for c in city_set}

    # ---- Pass 2: parse rows ----
    for idx, row in df.iterrows():
        data = row.get(data_col)
        if pd.isna(data) or not str(data).strip():
            continue

        data = str(data)

        # Extract zip codes
        zips = re.findall(r"\b(\d{5})(?:-\d{4})?\b", data)
        if len(zips) >= 1:
            df.at[idx, "DeathZip"] = zips[0]
        if len(zips) >= 2:
            df.at[idx, "EventZip"] = zips[1]

        # Split into sections around the zip codes
        no_zips = re.sub(r"\b\d{5}(?:-\d{4})?\b", "|", data)
        parts = no_zips.split("|")

        # DeathCity is everything before the first zip
        if len(parts) >= 1 and parts[0].strip():
            death_city = norm(parts[0])
            if death_city and death_city.upper() != "NULL":
                df.at[idx, "DeathCity"] = death_city

        # Event section is between first and second zip (when present)
        if len(parts) >= 2 and parts[1].strip():
            event_section = norm(parts[1])

            # Handle NULL-only rows
            if event_section.upper().startswith("NULL"):
                # leave EventPlace/Address/City as NULL
                continue

            tokens = event_section.split()
            if not tokens:
                continue

            event_place = tokens[0]
            df.at[idx, "EventPlace"] = event_place

            rest = norm(" ".join(tokens[1:]))

            if not rest:
                continue

            # If rest ends in a street suffix (e.g., "Sherman Way"), it's probably address-only (no city)
            last_word = rest.split()[-1].lower()
            if last_word in street_suffixes:
                df.at[idx, "EventAddress"] = rest
                df.at[idx, "EventCity"] = "NULL"
                continue

            # If rest is exactly a known city, then it’s city-only (no address)
            if rest.lower() in city_set_ci:
                df.at[idx, "EventCity"] = city_set_ci[rest.lower()]
                df.at[idx, "EventAddress"] = "NULL"
                continue

            # Try to split address vs city by matching the LONGEST city suffix (up to 4 tokens)
            rest_tokens = rest.split()
            found_city = None
            found_n = 0
            max_n = min(4, len(rest_tokens))

            for n in range(max_n, 0, -1):
                cand = " ".join(rest_tokens[-n:])
                if cand.lower() in city_set_ci:
                    found_city = city_set_ci[cand.lower()]
                    found_n = n
                    break

            if found_city:
                addr = norm(" ".join(rest_tokens[:-found_n]))
                df.at[idx, "EventCity"] = found_city
                df.at[idx, "EventAddress"] = addr if addr else "NULL"
            else:
                # Fallback:
                # - if it starts with a number, treat as address (city unknown)
                # - otherwise treat as city-ish free text
                if re.match(r"^\d", rest):
                    df.at[idx, "EventAddress"] = rest
                    df.at[idx, "EventCity"] = "NULL"
                else:
                    df.at[idx, "EventCity"] = rest
                    df.at[idx, "EventAddress"] = "NULL"

        # Special case: sometimes there is no second zip and the event section can drift into parts[2]
        # (you can add handling here if you see those cases in output)

    return df


def parse_schema2_data(df: pd.DataFrame, data_col: str) -> pd.DataFrame:
    """Parse schema 2 (page 3) data - Injury, Mode, AND CauseA."""
    for idx, row in df.iterrows():
        data = row[data_col]
        if pd.isna(data) or not data:
            continue

        modes = ["ACCIDENT", "NATURAL", "SUICIDE", "HOMICIDE", "UNDETERMINED", "PENDING"]
        mode_found = None
        mode_pos = -1
        data_upper = data.upper()

        for mode in modes:
            pos = data_upper.rfind(mode)
            if pos != -1:
                # Ensure it's a whole word
                if (pos == 0 or not data_upper[pos-1].isalnum()) and \
                   (pos + len(mode) == len(data_upper) or not data_upper[pos + len(mode)].isalnum()):
                    mode_found = mode
                    mode_pos = pos
                    break
        
        if mode_found:
            df.at[idx, "Mode"] = mode_found
            
            # Everything before mode is injury description
            injury_text = data[:mode_pos].strip()
            df.at[idx, "InjuryDesc"] = injury_text if injury_text else "NULL"
            
            # Everything after mode is DeathCauseA
            cause_a_text = data[mode_pos + len(mode_found):].strip()
            df.at[idx, "DeathCauseA"] = cause_a_text if cause_a_text else "NULL"
            
        else:
            # No mode found, logic from before
            if "NULL" in data.upper():
                parts = data.upper().split("NULL")
                if parts[0].strip():
                    df.at[idx, "InjuryDesc"] = data[: len(parts[0])].strip()
                df.at[idx, "Mode"] = "NATURAL"
            else:
                df.at[idx, "InjuryDesc"] = data
                df.at[idx, "Mode"] = "NATURAL"
            
            df.at[idx, "DeathCauseA"] = "NULL" # Ensure it's set

    return df

def parse_schema3_data(df: pd.DataFrame, data_col: str) -> pd.DataFrame:
    """Parse schema 3 (page 4) data - Death Causes A and B."""
    for idx, row in df.iterrows():
        data = row[data_col]
        if pd.isna(data) or not data:
            continue

        # Look for NULL as separator
        if "NULL" in data:
            parts = data.split("NULL", 1)
            cause_a = parts[0].strip()

            if cause_a:
                df.at[idx, "DeathCauseA"] = cause_a
            else:
                df.at[idx, "DeathCauseA"] = "NULL"

            # Usually DeathCauseB is NULL when NULL separator is present
            df.at[idx, "DeathCauseB"] = "NULL"
        else:
            # No NULL separator - look for natural breaks
            # Common patterns that indicate cause B
            cause_b_keywords = [
                "DUE TO",
                "SECONDARY TO",
                "WITH",
                "COMPLICATED BY",
                "CAUSED BY",
                "AS A CONSEQUENCE OF",
                "RESULTING FROM",
            ]

            separator_found = False
            upper_data = data.upper()

            for keyword in cause_b_keywords:
                if keyword in upper_data:
                    keyword_pos = upper_data.find(keyword)
                    df.at[idx, "DeathCauseA"] = data[:keyword_pos].strip()
                    # Include the keyword in cause B
                    df.at[idx, "DeathCauseB"] = data[keyword_pos:].strip()
                    separator_found = True
                    break

            if not separator_found:
                # Check if there are multiple distinct medical conditions
                # (usually separated by commas or semicolons)
                if ";" in data:
                    parts = data.split(";", 1)
                    df.at[idx, "DeathCauseA"] = parts[0].strip()
                    df.at[idx, "DeathCauseB"] = parts[1].strip()
                else:
                    # Entire text is cause A
                    df.at[idx, "DeathCauseA"] = data
                    df.at[idx, "DeathCauseB"] = "NULL"

    return df


def parse_schema4_data(df: pd.DataFrame, data_col: str) -> pd.DataFrame:
    """Parse schema 4 (page 5) data - Death Causes B, C, and D."""
    for idx, row in df.iterrows():
        data = row[data_col]
        if pd.isna(data) or not data or data.upper() == "NULL":
            continue

        # Data is likely 'CauseB Text CauseC Text CauseD Text'
        # This is very hard to parse reliably.
        # We will look for "NULL" as a separator first.
        
        data = data.strip()
        if "NULL" in data:
            parts = data.split("NULL")
            parts = [p.strip() for p in parts if p.strip()] # Get non-empty parts
            
            if len(parts) >= 1:
                df.at[idx, "DeathCauseB"] = parts[0]
            if len(parts) >= 2:
                df.at[idx, "DeathCauseC"] = parts[1]
            if len(parts) >= 3:
                df.at[idx, "DeathCauseD"] = parts[2]
        else:
            # No NULLs. This is the hard case.
            # We'll put everything in CauseB for now.
            # A more complex parser would be needed to split this.
            # Example: "STAGE IV LIVER CANCER ALCOHOLIC LIVER CIRRHOSIS TYPE II DIABETES"
            # It's not feasible to split this with simple rules.
            df.at[idx, "DeathCauseB"] = data
            df.at[idx, "DeathCauseC"] = "NULL"
            df.at[idx, "DeathCauseD"] = "NULL"
            
    return df


def parse_schema5_data(df: pd.DataFrame, data_col: str) -> pd.DataFrame:
    """Parse schema 5 (page 6) data - Other Cause."""
    for idx, row in df.iterrows():
        data = row[data_col]
        if pd.isna(data) or not data or data.upper() == "NULL":
            df.at[idx, "OtherCause"] = "NULL"
        else:
            # Clean up the text
            cleaned = data.strip()
            # Remove multiple spaces
            cleaned = re.sub(r"\s+", " ", cleaned)
            df.at[idx, "OtherCause"] = cleaned

    return df


def parse_schema6_data(df: pd.DataFrame, data_col: str) -> pd.DataFrame:
    """Parse schema 6 (page 7) data - Demographics (Race, Gender, Age)."""
    for idx, row in df.iterrows():
        data = row[data_col]
        if pd.isna(data) or not data:
            continue

        # Extract age first (most reliable pattern)
        age_patterns = [
            r"(\d+)\s*years?\s*old",
            r"(\d+)\s*years?",
            r"(\d+)\s*y/?o",
            r"age\s*:?\s*(\d+)",
            r"(\d+)\s*yr",
        ]

        age_found = False
        for pattern in age_patterns:
            age_match = re.search(pattern, data, re.IGNORECASE)
            if age_match:
                df.at[idx, "Age"] = age_match.group(1) + " years"
                # Remove age from text for easier parsing
                data = data[: age_match.start()] + " " + data[age_match.end() :]
                age_found = True
                break

        # Extract gender
        if re.search(r"\bMale\b", data, re.IGNORECASE):
            df.at[idx, "Gender"] = "Male"
            data = re.sub(r"\bMale\b", "", data, flags=re.IGNORECASE)
        elif re.search(r"\bFemale\b", data, re.IGNORECASE):
            df.at[idx, "Gender"] = "Female"
            data = re.sub(r"\bFemale\b", "", data, flags=re.IGNORECASE)
        elif re.search(r"\b[MF]\b", data):
            # Single letter gender
            if re.search(r"\bM\b", data):
                df.at[idx, "Gender"] = "Male"
                data = re.sub(r"\bM\b", "", data)
            else:
                df.at[idx, "Gender"] = "Female"
                data = re.sub(r"\bF\b", "", data)

        # Remaining text is race/ethnicity
        remaining = data.strip()
        remaining = re.sub(r"\s+", " ", remaining)  # Clean multiple spaces

        if remaining and remaining.upper() != "NULL":
            # Map common variations
            race_mappings = {
                "W": "White/Caucasian",
                "B": "Black",
                "H": "Hispanic/Latino",
                "A": "Asian",
                "Hispanic": "Hispanic/Latino",
                "Latino": "Hispanic/Latino",
                "White": "White/Caucasian",
                "Caucasian": "White/Caucasian",
                "AA": "Black",
                "African American": "Black",
                "Black/African American": "Black",
                "Asian/Pacific Islander": "Asian",
                "Native American": "Native American",
                "Other": "Other",
            }

            # Check for exact matches first
            if remaining in race_mappings:
                df.at[idx, "Races"] = race_mappings[remaining]
            else:
                # Check for partial matches
                race_found = False
                for abbrev, full in race_mappings.items():
                    if abbrev.lower() in remaining.lower():
                        df.at[idx, "Races"] = full
                        race_found = True
                        break

                if not race_found:
                    # Use as is
                    df.at[idx, "Races"] = remaining

    return df


# Additional utility function to clean final dataframe
def clean_final_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleanup of the dataframe.
    """
    # Remove schema data columns
    schema_cols = [
        col for col in df.columns if col.startswith("Schema") and "_Data" in col
    ]
    page_cols = [
        col for col in df.columns if col.startswith("Schema") and "_Pages" in col
    ]

    # Keep page information in a separate column if needed
    if page_cols:
        df["SourcePages"] = df[page_cols].apply(
            lambda x: ",".join(x.dropna().astype(str)), axis=1
        )

    # Drop schema columns
    df = df.drop(columns=schema_cols + page_cols, errors="ignore")

    # Ensure all expected columns are present
    expected_columns = [
        "CaseNum",
        "FirstName",
        "LastName",
        "BirthDate",
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
    ]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = "NULL"

    # Reorder columns
    df = df[
        expected_columns + [col for col in df.columns if col not in expected_columns]
    ]

    # Final cleaning
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("NULL")
            df[col] = df[col].replace("", "NULL")
            df[col] = df[col].replace("nan", "NULL")

    return df


@app.command()
def main(
    pdf_path: str = typer.Argument(
        "./pipeline_steps/input_files/raw/PRARequest12102025.pdf",
        help="Path to the input PDF file",
    ),
    output_csv: str = typer.Argument(
        "./pipeline_steps/input_files/converted/prarequest121025.csv",
        help="Path to the output CSV file",
    ),
):

    # Step 1: Analyze format
    print("Analyzing PDF format...")
    analysis = analyze_pdf_format(pdf_path)  # Analyze more pages

    print(f"\nFormat Analysis:")
    print(f"  Pages with tables: {analysis['table_pages']}")
    print(f"  Pages with text only: {analysis['text_pages']}")
    print(f"  Empty pages: {analysis['empty_pages']}")
    print(f"  Case number patterns: {dict(analysis['case_number_patterns'])}")

    if analysis["problem_pages"]:
        print(f"\n  Problem pages found: {len(analysis['problem_pages'])}")
        for prob in analysis["problem_pages"][:5]:
            print(f"    Page {prob['page']}: {prob['issue']}")

    # Step 2: Extract data
    print("\nExtracting data...")
    df = simple_extraction_with_fallback(pdf_path)

    if not df.empty:
        # Step 3: Post-process
        print("\nPost-processing data...")
        df = post_process_raw_data(df)

        # Step 4: Final cleanup
        df = clean_final_dataframe(df)

        df.rename(
            columns={
                "DeathCauseA": "CauseA",
                "DeathCauseB": "CauseB",
                "DeathCauseC": "CauseC",
                "DeathCauseD": "CauseD",
            },
            inplace=True,
        )

        # Save
        df.to_csv(output_csv, index=False)

        print(f"\nExtraction complete!")
        print(f"  Total cases: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Saved to: {output_csv}")

        # Show sample
        print("\nFirst 5 rows:")
        print(df.head())

        # Show data completeness
        print("\nData completeness:")
        null_counts = df.eq("NULL").sum()
        for col in df.columns:
            non_null = len(df) - null_counts[col]
            pct = (non_null / len(df)) * 100
            print(f"  {col}: {non_null}/{len(df)} ({pct:.1f}% complete)")
    else:
        print("\nNo data extracted. Check the debug output above.")


# Main execution
if __name__ == "__main__":
    app()
