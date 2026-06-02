#!/usr/bin/env python
# coding: utf-8
"""
nflis_classifier.py
===================
Apply NFLIS-backed regex substance classification to an overdose CSV.

This tool supplements and corrects BERT-classified substance columns:

  1. Regex supplement — runs the classifier over all six cause-of-death text
     fields (CauseA-D, CauseOther, HowInjuryOccurred).  The BERT model only
     saw the `text` field (CauseA-D concatenated) and misses substance mentions
     in CauseOther and HowInjuryOccurred.  New detections are OR-combined with
     BERT's flags for all eight substance columns, so no true positive is lost.

  2. BERT false-positive correction — MDMA mis-labelled as Methamphetamine:
     BERT fires on the substring "methamphetamine" inside
     "methylenedioxymethamphetamine".  Fix: zero out Methamphetamine for
     BERT-only records whose cause text contains an MDMA term (the regex
     classifier correctly routes these to Others).

MODES
-----
  apply   Read --input CSV, apply corrections, write --output CSV.
  diff    Read --input CSV, apply corrections in memory, write rows that changed
          to --output CSV (or print a per-column summary if --output is omitted).
          Each changed row includes the matched_evidence column showing which
          text field and term triggered the change — useful for manual review.

USAGE
-----
  python nflis_classifier.py apply --input data/classified.csv --output data/corrected.csv
  python nflis_classifier.py diff  --input data/classified.csv --output data/changes.csv
  python nflis_classifier.py diff  --input data/classified.csv   # prints summary only
"""

import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default paths (override with CLI flags)
# ---------------------------------------------------------------------------
ROOT              = Path(__file__).resolve().parent.parent
DEFAULT_NFLIS     = ROOT / "data" / "NFLIS-Substances_Excel_20260520-222324728.csv"

# ---------------------------------------------------------------------------
# Substance columns handled by this classifier
# ---------------------------------------------------------------------------
SUBSTANCE_COLS = [
    "Methamphetamine", "Heroin", "Cocaine", "Fentanyl",
    "Alcohol", "Prescription.opioids", "Benzodiazepines", "Others",
]

# All six cause-of-death text fields searched by the regex.
# BERT only ever saw `text` (= CauseA-D concatenated).
SEARCH_FIELDS = [
    "CauseA", "CauseB", "CauseC", "CauseD", "CauseOther", "HowInjuryOccurred",
]

# ---------------------------------------------------------------------------
# NFLIS category → substance column
# ---------------------------------------------------------------------------
NFLIS_CATEGORY_MAP = {
    "Phenethylamines (Amphetamines) (S)":  "Methamphetamine",
    "Heroin":                              "Heroin",
    "Cocaine Alkaloids":                   "Cocaine",
    "Fentanyl and Fentanyl-related":       "Fentanyl",
    "Benzodiazepines":                     "Benzodiazepines",
    "Narcotic Analgesics":                 "Prescription.opioids",
    # Cannabinoids, psychedelics, novel psychoactives → Others
    "Cannabinoids":                        "Others",
    "Synthetic Cannabinoids":              "Others",
    "Synthetic Cathinones":                "Others",
    "Synthetic Cathinones (Hallucinogen)": "Others",
    "Hallucinogens":                       "Others",
    "Tryptamines":                         "Others",
    "Psychedelics":                        "Others",
    "Piperazines (Hallucinogen)":          "Others",
    "Piperazines (Stimulant)":             "Others",
    "Piperidines/PEA":                     "Others",
    "Phenethylamines":                     "Others",
    "Phenethylamines (Amphetamines) (H)":  "Others",
    "Phenethylamines (2C Series) (H)":     "Others",
    "Phenethylamines (Aminoindanes) (H)":  "Others",
    "Phenethylamines (D Series) (H)":      "Others",
    "Phenethylamines (Other) (H)":         "Others",
    "Phenethylamines (Other) (S)":         "Others",
    "Phenethylamines (Stimulant)":         "Others",
    "Depressants and Tranquilizers":       "Others",   # barbiturates, PCP, ketamine
    "Stimulants":                          "Others",   # non-amphetamine stimulants
    # Skipped: Analgesics (NSAIDs), Antidepressants, Steroids, Other, New Drug
}

# Narcotics category contains mixed signals; route per substance.
# Antagonists (naloxone, naltrexone) are intentionally excluded.
NARCOTICS_OVERRIDE = {
    "benzoylecgonine":  "Cocaine",             # cocaine metabolite
    "ecgonine":         "Cocaine",
    "norcocaine":       "Cocaine",
    "cocaethylene":     "Cocaine",
    "6-acetylmorphine": "Heroin",              # heroin-specific metabolite
    "3-acetylmorphine": "Heroin",
    "acetylmorphine":   "Heroin",
    "morphine":         "Prescription.opioids",
    "codeine":          "Prescription.opioids",
    "dihydrocodeine":   "Prescription.opioids",
    "noscapine":        "Prescription.opioids",
    "papaverine":       "Prescription.opioids",
    "thebaine":         "Prescription.opioids",
    "oripavine":        "Prescription.opioids",
    "opium":            "Prescription.opioids",
    "diphenoxylate":    "Prescription.opioids",
}

# ---------------------------------------------------------------------------
# Hard-coded patterns: street names, abbreviations, clinical shorthand absent
# from NFLIS but common in coroner text.  Each entry: (pattern_str, column).
# All use \b word-boundary anchors.
# Removed due to high false-positive rates:
#   "ice"   → fires on "ice cream", "black ice" (6 FPs)
#   "speed" → fires on "high rate of speed", traffic deaths (16 FPs)
#   "weed"  → fires on "weed puller" (1 FP)
#   "pot"   → fires on "clay pot", "pot of boiling liquid" (2 FPs)
#   "acid"  → fires on "acid fast bacilli" (TB), "acid reflux", "sulfuric acid" (many FPs)
# ---------------------------------------------------------------------------
ESSENTIAL_PATTERNS = [
    # --- Methamphetamine ---
    (r"\bmeth\b",                  "Methamphetamine"),
    (r"\bcrystal\s+meth\b",        "Methamphetamine"),
    (r"\bmathamphetamine\b",       "Methamphetamine"),
    (r"\bmetamphetamine\b",        "Methamphetamine"),
    (r"\bmetahmphetamine\b",       "Methamphetamine"),
    (r"\bmetehamphetamine\b",      "Methamphetamine"),
    (r"\bmetethmphetamine\b",      "Methamphetamine"),
    (r"\bdextroamphetamine\b",     "Methamphetamine"),
    (r"\badderall\b",              "Methamphetamine"),
    (r"\britalin\b",               "Methamphetamine"),
    (r"\bmethylphenidate\b",       "Methamphetamine"),
    (r"\bvyvanse\b",               "Methamphetamine"),
    (r"\blisdexamfetamine\b",      "Methamphetamine"),
    # --- Heroin ---
    (r"\bdiacetylmorphine\b",      "Heroin"),
    (r"\bdiamorphine\b",           "Heroin"),
    (r"\bheron\b",                 "Heroin"),
    (r"\b6[\s-]?mam\b",            "Heroin"),   # 6-monoacetylmorphine
    (r"\bsmack\b",                 "Heroin"),
    (r"\bjunk\b",                  "Heroin"),
    (r"\bbrown\s+sugar\b",         "Heroin"),
    (r"\bblack\s+tar\b",           "Heroin"),
    # --- Cocaine ---
    (r"\bcrack\b",                 "Cocaine"),
    (r"\bfreebase\b",              "Cocaine"),
    (r"\bcoke\b",                  "Cocaine"),
    (r"\bcocane\b",                "Cocaine"),
    (r"\bbenzoylecgonine\b",       "Cocaine"),
    (r"\becgonine\b",              "Cocaine"),
    (r"\bcocaethylene\b",          "Cocaine"),
    # --- Fentanyl analogues ---
    (r"\bcarfentanil\b",           "Fentanyl"),
    (r"\bacetylfentanyl\b",        "Fentanyl"),
    (r"\bbutyrylfentanyl\b",       "Fentanyl"),
    (r"\bfuranylfentanyl\b",       "Fentanyl"),
    (r"\bacryl\s*fentanyl\b",      "Fentanyl"),
    # --- Alcohol ---
    (r"\bethanol\b",               "Alcohol"),
    (r"\betoh\b",                  "Alcohol"),
    (r"\balcohol\b",               "Alcohol"),
    (r"\balcoholic\b",             "Alcohol"),
    (r"\balcoholism\b",            "Alcohol"),
    (r"\bchronic\s+alcohol\b",     "Alcohol"),
    (r"\balcoholic\s+cirrhosis\b", "Alcohol"),
    (r"\blaennec\b",               "Alcohol"),   # Laennec's cirrhosis = alcoholic cirrhosis
    # --- Prescription opioids ---
    (r"\bopioid\b",                "Prescription.opioids"),
    (r"\bopioids\b",               "Prescription.opioids"),
    (r"\bopiate\b",                "Prescription.opioids"),
    (r"\bopiates\b",               "Prescription.opioids"),
    (r"\bopium\b",                 "Prescription.opioids"),
    (r"\bmorphine\b",              "Prescription.opioids"),
    (r"\bcodeine\b",               "Prescription.opioids"),
    (r"\boxycodone\b",             "Prescription.opioids"),
    (r"\bhydrocodone\b",           "Prescription.opioids"),
    (r"\bmethadone\b",             "Prescription.opioids"),
    (r"\bbuprenorphine\b",         "Prescription.opioids"),
    (r"\bsuboxone\b",              "Prescription.opioids"),
    (r"\bsubutex\b",               "Prescription.opioids"),
    (r"\btramadol\b",              "Prescription.opioids"),
    (r"\bhydromorphone\b",         "Prescription.opioids"),
    (r"\bdilaudid\b",              "Prescription.opioids"),
    (r"\boxymorphone\b",           "Prescription.opioids"),
    (r"\bopana\b",                 "Prescription.opioids"),
    (r"\bvicodin\b",               "Prescription.opioids"),
    (r"\bpercocet\b",              "Prescription.opioids"),
    (r"\bnorco\b",                 "Prescription.opioids"),
    (r"\btapentadol\b",            "Prescription.opioids"),
    (r"\bmeperidine\b",            "Prescription.opioids"),
    (r"\bdemerol\b",               "Prescription.opioids"),
    (r"\bpropoxyphene\b",          "Prescription.opioids"),
    (r"\bdarvon\b",                "Prescription.opioids"),
    (r"\bkratom\b",                "Prescription.opioids"),
    (r"\bmitragynine\b",           "Prescription.opioids"),
    (r"\blevorphanol\b",           "Prescription.opioids"),
    (r"\bpentazocine\b",           "Prescription.opioids"),
    (r"\bbutorphanol\b",           "Prescription.opioids"),
    (r"\bnalbuphine\b",            "Prescription.opioids"),
    (r"\bdesomorphine\b",          "Prescription.opioids"),   # krokodil
    (r"\bkrokodil\b",              "Prescription.opioids"),
    (r"\bu-?47700\b",              "Prescription.opioids"),   # novel synthetic opioid
    (r"\bisotonitazene\b",         "Prescription.opioids"),
    (r"\betonitazene\b",           "Prescription.opioids"),
    (r"\bmetonitazene\b",          "Prescription.opioids"),
    (r"\bprotonitazene\b",         "Prescription.opioids"),
    (r"\bnitazene\b",              "Prescription.opioids"),
    # --- Benzodiazepines ---
    (r"\bbenzodiazepine\b",        "Benzodiazepines"),
    (r"\bbenzodiazepines\b",       "Benzodiazepines"),
    (r"\bbenzo\b",                 "Benzodiazepines"),
    (r"\bbenzos\b",                "Benzodiazepines"),
    (r"\bdiazepam\b",              "Benzodiazepines"),
    (r"\bvalium\b",                "Benzodiazepines"),
    (r"\balprazolam\b",            "Benzodiazepines"),
    (r"\bxanax\b",                 "Benzodiazepines"),
    (r"\blorazepam\b",             "Benzodiazepines"),
    (r"\bativan\b",                "Benzodiazepines"),
    (r"\bclonazepam\b",            "Benzodiazepines"),
    (r"\bklonopin\b",              "Benzodiazepines"),
    (r"\btemazepam\b",             "Benzodiazepines"),
    (r"\brestoril\b",              "Benzodiazepines"),
    (r"\bchlordiazepoxide\b",      "Benzodiazepines"),
    (r"\blibrium\b",               "Benzodiazepines"),
    (r"\bmidazolam\b",             "Benzodiazepines"),
    (r"\bversed\b",                "Benzodiazepines"),
    (r"\btriazolam\b",             "Benzodiazepines"),
    (r"\bhalcion\b",               "Benzodiazepines"),
    (r"\bflurazepam\b",            "Benzodiazepines"),
    (r"\bnitrazepam\b",            "Benzodiazepines"),
    (r"\bflunitrazepam\b",         "Benzodiazepines"),
    (r"\brohypnol\b",              "Benzodiazepines"),
    (r"\bphenazepam\b",            "Benzodiazepines"),
    (r"\bbromazepam\b",            "Benzodiazepines"),
    (r"\boxazepam\b",              "Benzodiazepines"),
    (r"\bclobazam\b",              "Benzodiazepines"),
    (r"\bclonazolam\b",            "Benzodiazepines"),
    (r"\bflualprazolam\b",         "Benzodiazepines"),
    (r"\betizolam\b",              "Benzodiazepines"),
    (r"\bbrotizolam\b",            "Benzodiazepines"),
    # --- Others (NFLIS-defined: cannabinoids, psychedelics, novel psychoactives) ---
    (r"\bcannabis\b",              "Others"),
    (r"\bmarijuana\b",             "Others"),
    (r"\bthc\b",                   "Others"),
    (r"\bhashish\b",               "Others"),
    (r"\bcannabinoid\b",           "Others"),
    (r"\bcbd\b",                   "Others"),
    (r"\bdronabinol\b",            "Others"),
    (r"\bmdma\b",                  "Others"),
    (r"\becstasy\b",               "Others"),
    (r"\bmolly\b",                 "Others"),
    (r"\bmda\b",                   "Others"),
    (r"\bpcp\b",                   "Others"),
    (r"\bphencyclidine\b",         "Others"),
    (r"\bketamine\b",              "Others"),
    (r"\bspecial\s+k\b",           "Others"),
    (r"\bghb\b",                   "Others"),
    (r"\bgamma.?hydroxybutyrate\b","Others"),
    (r"\bgamma.?hydroxybutyric\b", "Others"),
    (r"\blsd\b",                   "Others"),
    (r"\blysergic\b",              "Others"),
    (r"\bpsilocybin\b",            "Others"),
    (r"\bpsilocin\b",              "Others"),
    (r"\bmagic\s+mushroom\b",      "Others"),
    (r"\bbath\s+salts\b",          "Others"),
    (r"\bspice\b",                 "Others"),
    (r"\bk2\b",                    "Others"),
    (r"\bxylazine\b",              "Others"),
    (r"\btranq\b",                 "Others"),
    (r"\bgabapentin\b",            "Others"),
    (r"\bpregabalin\b",            "Others"),
    (r"\bneurontin\b",             "Others"),
    (r"\bcatnip\b",                "Others"),
    (r"\bsalvia\b",                "Others"),
    (r"\bdmt\b",                   "Others"),
    (r"\bbarbiturate\b",           "Others"),
    (r"\bphenobarbital\b",         "Others"),
    (r"\bamobarbital\b",           "Others"),
    (r"\bsecobarbital\b",          "Others"),
    (r"\bbutalbital\b",            "Others"),
    (r"\bchoral\s+hydrate\b",      "Others"),
    (r"\bnitrous\s+oxide\b",       "Others"),
    (r"\bkhat\b",                  "Others"),
]

# Generic death-certificate phrases that establish a drug death without naming
# a classifiable substance. These set Any Drugs only; they should not be placed
# in the NFLIS substance vocabulary or routed to Others.
GENERAL_DRUG_PATTERNS = [
    r"\b(?:combined\s+)?effects?\s+of\s+(?:multiple\s+)?(?:drugs?|medications?)\b",
    r"\b(?:multiple\s+)?drugs?\s+(?:intoxication|toxicity|effects?)\b",
    r"\bmultiple\s+medications?\s+(?:intoxication|toxicity|effects?)\b",
    r"\bprescription\s+drug\s+overuse\b",
    r"\btook\s+sedative\s+medications\b",
    r"\b(?:unknown\s+route\s+of\s+)?drugs?\s+intake\b",
    r"\b(?:drug|medication)s?\s+intake\b",
    r"\bdrug/medication\s+intake\b",
    r"\bintake\s+of\s+(?:of\s+)?(?:the\s+)?(?:drug(?:\(s\))?|drugs?|medications?)\b",
    r"\binatke\s+of\s+(?:the\s+)?drugs?\b",
    r"\bingestion\s+of\s+(?:the\s+)?drugs?\b",
    r"\bdrug\s+abuse\b",
    r"\bdrug\s+use\b",
    r"\bdrug/medication\s+use\b",
    r"\bsubstance\s+use\b",
    r"\bsubstanceu\s*se\b",
    r"\binhalation\s+of\s+substance\b",
    r"\bsequelae\s+of\s+drug\s+use\b",
    r"\bsequeale\s+of\s+drug\s+use\b",
    r"\bconsequences\s+of\s+drug\s+abuse\b",
    r"\boverdose\s+of\s+unkn[o0]wn\s+drug\b",
    r"\bchronic\s+intravenous\s+narcotism\b",
]

# Pattern for detecting MDMA in cause-of-death text (used for Meth FP correction)
MDMA_PAT = re.compile(r"methylenedioxy|(?<!\w)mdma(?!\w)|ecstasy", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Pattern building
# ---------------------------------------------------------------------------

def _escape_term(term: str) -> str:
    """Escape a raw substance name for use in a word-boundary regex."""
    escaped = re.escape(term.strip())
    return re.sub(r"\\ ", r"\\s+", escaped)  # allow flexible whitespace


def build_patterns(nflis_path: Path = DEFAULT_NFLIS) -> dict:
    """
    Build compiled regex patterns per classification column from three sources:
      1. NFLIS substance names and synonyms (loaded from nflis_path)
      2. Hard-coded ESSENTIAL_PATTERNS (street names, clinical terms)
      3. Generic drug-death phrases that set Any Drugs only
    Returns {column_name: compiled_re.Pattern}.
    """
    nflis = pd.read_csv(nflis_path, encoding="utf-8-sig")
    terms: dict[str, list[str]] = {col: [] for col in SUBSTANCE_COLS}

    for _, row in nflis.iterrows():
        col = NFLIS_CATEGORY_MAP.get(row["NFLIS Detailed Drug Category"])
        if col is None:
            continue
        name = str(row["Substance Name"]).strip()
        if name and name.lower() not in ("nan", ""):
            terms[col].append(name)
        if pd.notna(row["Synonyms"]):
            for syn in str(row["Synonyms"]).split(";"):
                s = syn.strip()
                if s and s.lower() not in ("nan", ""):
                    terms[col].append(s)

    for _, row in nflis[nflis["NFLIS Detailed Drug Category"] == "Narcotics"].iterrows():
        col = NARCOTICS_OVERRIDE.get(str(row["Substance Name"]).strip().lower())
        if col:
            terms[col].append(str(row["Substance Name"]).strip())

    essential_by_col: dict[str, list[str]] = {col: [] for col in SUBSTANCE_COLS}
    for pattern_str, col in ESSENTIAL_PATTERNS:
        essential_by_col[col].append(pattern_str)

    compiled = {}
    for col in SUBSTANCE_COLS:
        nflis_parts = [
            r"\b" + _escape_term(t) + r"\b"
            for t in terms[col]
            if len(t) >= 3  # skip very short terms to avoid false positives
        ]
        all_parts = nflis_parts + essential_by_col[col]
        seen: set[str] = set()
        unique = [p for p in all_parts if not (p in seen or seen.add(p))]
        if unique:
            compiled[col] = re.compile("|".join(unique), re.IGNORECASE)

    compiled["Any Drugs"] = re.compile("|".join(GENERAL_DRUG_PATTERNS), re.IGNORECASE)
    return compiled


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def normalize_text(value) -> str:
    """Return a cleaned, normalised string from a raw cell value."""
    if pd.isna(value):
        return ""
    s = str(value).replace("\n", " ").replace("\r", " ")
    return re.sub(r"\bNULL\b", " ", s, flags=re.IGNORECASE)


def _row_text(row: pd.Series) -> str:
    """Concatenate all six cause-of-death text fields for a single record."""
    parts = [normalize_text(row.get(f, "")) for f in SEARCH_FIELDS]
    return " | ".join(p for p in parts if p.strip())


# ---------------------------------------------------------------------------
# Per-field match evidence (useful for auditing and diff inspection)
# ---------------------------------------------------------------------------

def classify_record_detail(row: pd.Series, patterns: dict) -> dict:
    """
    Return per-field match evidence for a single record.

    Returns {column: [(field_name, matched_text), ...]} for every substance
    column, plus Any Drugs, that has at least one match. Searches each SEARCH_FIELD separately
    so the triggering field is identifiable (unlike _row_text which joins them).
    """
    detail: dict[str, list] = {col: [] for col in SUBSTANCE_COLS + ["Any Drugs"]}
    for field in SEARCH_FIELDS:
        text = normalize_text(row.get(field, ""))
        if not text.strip():
            continue
        for col, pat in patterns.items():
            m = pat.search(text)
            if m:
                detail[col].append((field, m.group()))
    return detail


# ---------------------------------------------------------------------------
# Core correction pipeline
# ---------------------------------------------------------------------------

def apply_corrections(df: pd.DataFrame, patterns: dict, verbose: bool = True) -> pd.DataFrame:
    """
    Return a corrected copy of df with updated substance columns.

    Stage 1 — Regex supplement:
        For all eight columns, new value = BERT | regex.
    Stage 2 — MDMA correction:
        Zero out Methamphetamine where BERT=1, regex=0, and text contains MDMA.
        (These 83 cases in the full dataset are MDMA mis-labelled by BERT;
        Others is already set to 1 by the regex in stage 1.)
    Stage 3 — Generic drug-death phrases:
        Set Any Drugs for broad phrases such as "drug use" or "multiple drug
        intoxication" without assigning a substance column.
    """
    out = df.copy()

    if verbose:
        print(f"Running regex classifier on {len(out):,} records …", flush=True)

    # Vectorised classification: build one combined text per row, then search
    all_text = out.apply(_row_text, axis=1)
    regex_flags: dict[str, np.ndarray] = {
        col: all_text.apply(lambda t, p=pat: 1 if p.search(t) else 0).values
        for col, pat in patterns.items()
    }
    any_drug_regex = regex_flags.get(
        "Any Drugs",
        np.zeros(len(out), dtype=int),
    )
    n_generic_any = int(any_drug_regex.sum())

    # Compute MDMA mask on the ORIGINAL BERT flags before any modification.
    # Condition: BERT fired (=1), regex did not (=0), text contains MDMA term.
    bert_meth  = out["Methamphetamine"].fillna(0).astype(int).values
    regex_meth = regex_flags["Methamphetamine"]
    mdma_mask  = (
        (bert_meth == 1) &
        (regex_meth == 0) &
        all_text.apply(lambda t: bool(MDMA_PAT.search(t))).values
    )
    n_mdma = int(mdma_mask.sum())

    if verbose:
        print(f"\nApplying corrections ({n_mdma} MDMA mis-labels to clear):")
        print(f"  {'Column':<25} {'before':>7}  {'Δ':>6}  {'after':>7}")
        print(f"  {'-'*52}")

    for col in SUBSTANCE_COLS:
        before = int(out[col].fillna(0).astype(int).sum())

        # Stage 1: OR-combine for all columns so no BERT true positive is lost.
        out[col] = (out[col].fillna(0).astype(int).values | regex_flags[col]).astype(int)
        note = ""

        after = int(out[col].sum())
        if verbose:
            sign = "+" if after >= before else ""
            print(f"  {col:<25} {before:>7,}  {sign}{after - before:>5,}  {after:>7,}{note}")

    # Stage 2: clear Methamphetamine for MDMA cases (applied after the OR-combine).
    out.loc[mdma_mask, "Methamphetamine"] = 0
    if verbose:
        print(f"\n  Methamphetamine: cleared {n_mdma} MDMA mis-labels")

    # Recompute derived columns
    subst = out[SUBSTANCE_COLS].fillna(0).astype(int)
    out["Number_Substances"] = subst.sum(axis=1)
    out["Polysubstance"]     = (out["Number_Substances"] > 1).astype(int)
    original_any = (
        out["Any Drugs"].fillna(0).astype(int).values
        if "Any Drugs" in out.columns
        else np.zeros(len(out), dtype=int)
    )
    out["Any Drugs"] = (
        (out["Number_Substances"] > 0).astype(int).values |
        original_any |
        any_drug_regex
    ).astype(int)
    if verbose:
        before_any = int(original_any.sum())
        after_any = int(out["Any Drugs"].sum())
        sign = "+" if after_any >= before_any else ""
        print(
            f"  {'Any Drugs':<25} {before_any:>7,}  "
            f"{sign}{after_any - before_any:>5,}  {after_any:>7,}"
        )
        print(f"  Generic drug-death phrase matches: {n_generic_any:,}")
    out["Any Opioids"]       = (
        subst[["Heroin", "Fentanyl", "Prescription.opioids"]].sum(axis=1) > 0
    ).astype(int)

    return out


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def build_diff(original: pd.DataFrame, corrected: pd.DataFrame,
               patterns: dict = None) -> pd.DataFrame:
    """
    Return the subset of rows where any substance column or Any Drugs changed between
    `original` and `corrected`.

    Output columns:
      - CaseNumber, DeathDate (if present) for row identification
      - changes: human-readable summary of what changed (e.g. "Cocaine: 0→1")
      - matched_evidence: which field/term triggered each new classification
        (only populated when `patterns` is supplied)
      - For each substance and Any Drugs: original value then <col>_new value side-by-side
      - The six cause-of-death text fields for manual inspection
      - All remaining original columns
    """
    compare_cols = SUBSTANCE_COLS + ["Any Drugs"]
    changed = pd.Series(False, index=original.index)
    for col in compare_cols:
        if col in original.columns and col in corrected.columns:
            changed |= (
                original[col].fillna(0).astype(int) !=
                corrected[col].fillna(0).astype(int)
            )

    if not changed.any():
        return pd.DataFrame()

    diff = original[changed].copy()

    for col in compare_cols:
        if col in corrected.columns:
            diff[f"{col}_new"] = corrected.loc[changed, col].values

    def _summary(idx):
        parts = []
        for col in compare_cols:
            o = int(original.at[idx, col]) if col in original.columns else 0
            n = int(corrected.at[idx, col]) if col in corrected.columns else 0
            if o != n:
                parts.append(f"{col}: {o}→{n}")
        return "; ".join(parts)

    diff["changes"] = [_summary(idx) for idx in diff.index]

    # Per-row match evidence — shows which field and term drove each new flag
    if patterns is not None:
        evidence_parts = []
        for idx in diff.index:
            row = original.loc[idx]
            detail = classify_record_detail(row, patterns)
            # Only report evidence for columns that actually changed 0→1
            parts = []
            for col in compare_cols:
                o = int(original.at[idx, col]) if col in original.columns else 0
                n = int(corrected.at[idx, col]) if col in corrected.columns else 0
                if n > o and detail[col]:
                    hits = "; ".join(f"{f}:'{m}'" for f, m in detail[col])
                    parts.append(f"{col}=[{hits}]")
                elif o > n:
                    # Decrease (e.g. MDMA correction or Others discard)
                    parts.append(f"{col}=cleared")
            evidence_parts.append(" | ".join(parts))
        diff["matched_evidence"] = evidence_parts

    # Column order optimised for manual review
    id_cols    = [c for c in ["CaseNumber", "DeathDate"] if c in diff.columns]
    meta_cols  = [c for c in ["changes", "matched_evidence"] if c in diff.columns]
    subst_cols = []
    for col in compare_cols:
        if col in diff.columns:          subst_cols.append(col)
        if f"{col}_new" in diff.columns: subst_cols.append(f"{col}_new")
    text_cols  = [c for c in SEARCH_FIELDS if c in diff.columns]
    rest       = [c for c in diff.columns
                  if c not in id_cols + meta_cols + subst_cols + text_cols]

    return diff[id_cols + meta_cols + subst_cols + text_cols + rest]


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_apply(args):
    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  {len(df):,} records, {len(df.columns)} columns")

    print(f"Building patterns from: {args.nflis}")
    patterns = build_patterns(Path(args.nflis))
    n_total = sum(len(p.pattern.split("|")) for p in patterns.values())
    print(f"  {n_total:,} patterns across {len(patterns)} classification columns")

    corrected = apply_corrections(df, patterns, verbose=True)

    corrected.to_csv(args.output, index=False)
    print(f"\nSaved → {args.output}")


def cmd_diff(args):
    print(f"Loading: {args.input}")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  {len(df):,} records")

    print(f"Building patterns from: {args.nflis}")
    patterns = build_patterns(Path(args.nflis))

    corrected = apply_corrections(df, patterns, verbose=False)
    diff_df   = build_diff(df, corrected, patterns=patterns)

    print(f"\n{len(diff_df):,} rows changed across classification columns:")
    for col in SUBSTANCE_COLS + ["Any Drugs"]:
        if col not in df.columns or col not in corrected.columns:
            continue
        n = (
            df[col].fillna(0).astype(int) !=
            corrected[col].fillna(0).astype(int)
        ).sum()
        if n:
            orig_total = int(df[col].fillna(0).astype(int).sum())
            new_total  = int(corrected[col].fillna(0).astype(int).sum())
            print(f"  {col:<25} {n:>5,} rows changed  "
                  f"(column total: {orig_total:,} → {new_total:,})")

    if args.output:
        diff_df.to_csv(args.output, index=False)
        print(f"\nSaved diff → {args.output}")
    else:
        print("\n(Pass --output <path.csv> to save the diff for inspection)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_apply = sub.add_parser("apply", help="Apply regex corrections to input CSV")
    p_apply.add_argument("--input",  required=True, help="Input BERT-classified CSV")
    p_apply.add_argument("--output", required=True, help="Output corrected CSV")
    p_apply.add_argument("--nflis",  default=str(DEFAULT_NFLIS),
                         help="NFLIS substances CSV (default: %(default)s)")

    p_diff = sub.add_parser("diff", help="Identify rows changed by corrections")
    p_diff.add_argument("--input",  required=True, help="Input BERT-classified CSV")
    p_diff.add_argument("--output", default=None,
                        help="Output diff CSV (omit to print summary only)")
    p_diff.add_argument("--nflis",  default=str(DEFAULT_NFLIS),
                        help="NFLIS substances CSV (default: %(default)s)")

    args = parser.parse_args()
    {"apply": cmd_apply, "diff": cmd_diff}[args.command](args)


if __name__ == "__main__":
    main()
