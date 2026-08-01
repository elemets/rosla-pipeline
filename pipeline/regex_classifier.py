#!/usr/bin/env python
# coding: utf-8
"""
nflis_classifier.py
===================
Apply NFLIS-backed regex substance classification to an overdose CSV.

This tool supplements and corrects BERT-classified substance columns:

  1. Regex supplement — runs the classifier over all six cause-of-death text
     fields (CauseA-D, CauseOther, HowInjuryOccurred), the same fields
     `classify.py::create_text_col` concatenates into the `text` field BERT
     reads.  BERT still misses substance mentions the patterns catch, so new
     detections are OR-combined with BERT's flags for all eight substance
     columns and no true positive is lost.

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
from collections import defaultdict
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

# All six cause-of-death text fields searched by the regex — the same fields
# `classify.py::create_text_col` concatenates into the `text` field BERT reads.
#
# The last two carry two spellings each. Raw coroner exports name them
# OtherCause / InjuryDesc; join_similar_columns renames them to CauseOther /
# HowInjuryOccurred, but that runs *after* classification. Listing both means the
# regex reads them whichever stage it is handed. Missing names resolve to "" in
# `_row_text`, so naming a spelling a file does not use costs nothing, and a file
# carrying both just repeats the text (harmless to a search).
SEARCH_FIELDS = [
    "CauseA", "CauseB", "CauseC", "CauseD",
    "CauseOther", "OtherCause",
    "HowInjuryOccurred", "InjuryDesc",
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

# NFLIS-loaded substance names that need to be routed to a different column
# than their category default, or dropped entirely (validated against
# data/train_set.csv: generic "amphetamine" mentions are Others=1 95% of the
# time, not Methamphetamine; "phentermine" shows no clean signal either way
# and is excluded rather than guessed at).
NFLIS_NAME_OVERRIDE = {
    "amphetamine": "Others",
    "mitragynine": "Others",
    "7-hydroxymitragynine": "Others",
    "mitragynine pseudoindoxyl": "Others",
}
# "phentermine": no clean signal for any column (0/2 in train_set.csv).
# "brorphine": no clean signal for Others or Prescription.opioids (~20% either way).
# "methorphan": ambiguous -- often a comma-tokenization artifact splitting
# "dextromethorphan"/"levomethorphan" into DEXTRO/LEVO/METHORPHAN tokens;
# NFLIS's standalone "Methorphan" entry can't be distinguished from that split.
# Nitazene-family designer opioids: no clean signal for Others or
# Prescription.opioids (~20-25% either way in train_set.csv).
NFLIS_NAME_EXCLUSIONS = {
    "phentermine", "brorphine", "methorphan",
    "isotonitazene", "etonitazene", "metonitazene", "protonitazene",
    "clonitazene", "butonitazene", "flunitazene",
    "5-methyl etodesnitazene", "etodesnitazene", "metodesnitazene",
    "protodesnitazene", "ethylene etonitazene", "ethyleneoxynitazene",
    "methylenedioxynitazene", "n,n-dimethylamino etonitazene",
    "n-desethyl isotonitazene", "n-piperidinyl etonitazene",
    "n-pyrrolidino ethylene isotonitazene", "n-pyrrolidino isotonitazene",
    "n-pyrrolidino etonitazene", "n-pyrrolidino metonitazene",
    "n-pyrrolidino protonitazene", "n-desethyl protonitazene",
    "n-desethyl etonitazene",
}
# NOTE: NFLIS also lists the bare class word "Benzodiazepine" as if it were
# a specific substance name (Substance Name row, category "Benzodiazepines").
# This looked like a 2% precision leak against train_set_v2.csv, but that
# turned out to be a train labeling gap (58 unambiguous "BENZODIAZEPINE
# TOXICITY"-style records with Benzodiazepines=0, since fixed) -- 100%
# precise after the fix, so it's deliberately NOT excluded here. See
# ESSENTIAL_PATTERNS below.

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
    (r"\bethanolism\b",            "Alcohol"),
    # --- Prescription opioids ---
    # "opioid"/"opioids"/"opiate"/"opiates" deliberately excluded here:
    # generic class words, not named substances -- a specific column should
    # only match a specific drug. Verified against train_set_v2.csv
    # (2026-07): precision against Prescription.opioids specifically is
    # catastrophic (0-17%, n=10-72 each) -- these were a pre-existing bug,
    # silently injecting false positives via the OR-combine correction
    # stage. All four terms ARE a 100% reliable signal for the Any Opioids
    # AGGREGATE column instead -- see GENERAL_OPIOID_PATTERNS.
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
    (r"\bkratom\b",                "Others"),
    (r"\bmitragynine\b",           "Others"),
    (r"\blevorphanol\b",           "Prescription.opioids"),
    (r"\bpentazocine\b",           "Prescription.opioids"),
    (r"\bbutorphanol\b",           "Prescription.opioids"),
    (r"\bnalbuphine\b",            "Prescription.opioids"),
    (r"\bdesomorphine\b",          "Prescription.opioids"),   # krokodil
    (r"\bkrokodil\b",              "Prescription.opioids"),
    (r"\bu-?47700\b",              "Prescription.opioids"),   # novel synthetic opioid
    # Nitazene-family designer opioids removed: train_set.csv shows no clean
    # signal for either Prescription.opioids or Others (~20-25% precision on
    # both), so we can't confidently route them without domain input.
    # --- Benzodiazepines ---
    # "benzodiazepine(s)" initially looked like the same generic-class-word
    # bug as Prescription.opioids (2-5% precision, n=21-48 in
    # train_set_v2.csv), but turned out to be a train labeling gap instead:
    # the low-precision records were unambiguous "BENZODIAZEPINE TOXICITY"-
    # style cause-of-death text, not negated or ambiguous. Corrected in
    # train_set_v2.csv (58 rows); 100% precise after the fix. Unlike
    # "opioid" there's no aggregate column to disambiguate to here
    # (Benzodiazepines is both the specific and only column for this
    # substance), so once the labels are right, the generic term maps
    # straight to it. "benzo"/"benzos" have zero occurrences in
    # train_set_v2.csv to verify either way, but are the same kind of
    # class-word shorthand, so included on the same principle.
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
    # NFLIS lists this family only with its ring-position prefix
    # ("3,4-Methylenedioxymethamphetamine"), which coroners routinely drop. The
    # bare spellings must match here or MDMA cases end up with no substance at
    # all: stage 2 clears the Methamphetamine that BERT fires on the embedded
    # "methamphetamine" substring, and nothing sets Others in its place.
    (r"\bmethylenedioxy(?:meth|ethyl)?amphetamine\b", "Others"),
    (r"\bmethylenedioxypyrovalerone\b",               "Others"),
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
    (r"\britalin\b",               "Others"),
    (r"\bmethylphenidate\b",       "Others"),
    (r"\bcitalopram\b",            "Others"),
    (r"\bescitalopram\b",          "Others"),
    (r"\bclozapine\b",             "Others"),
    (r"\btrazodone\b",             "Others"),
    (r"\bdiphenhydramine\b",       "Others"),
    (r"\bamphetamines\b",          "Others"),   # plural, NFLIS only auto-loads singular
    (r"\bolanzapine\b",            "Others"),
    (r"\bfluoxetine\b",            "Others"),
    (r"\bquetiapine\b",            "Others"),
    (r"\bsertraline\b",            "Others"),
    (r"\bduloxetine\b",            "Others"),
    (r"\bmirtazapine\b",           "Others"),
    (r"\bparoxetine\b",            "Others"),
    (r"\bvenlafaxine\b",           "Others"),
    (r"\btopiramate\b",            "Others"),
    (r"\bdoxylamine\b",            "Others"),
    (r"\blevetiracetam\b",         "Others"),
    (r"\baripiprazole\b",          "Others"),
    (r"\brisperidone\b",           "Others"),
    (r"\blamotrigine\b",           "Others"),
    (r"\bclonidine\b",             "Others"),
    (r"\bbupropion\b",             "Others"),
    (r"\bamitriptyline\b",         "Others"),
    (r"\bnortriptyline\b",         "Others"),
    (r"\bpaliperidone\b",          "Others"),
    # --- Non-psychiatric substances that ARE in the NFLIS reference file but sit
    # in categories NFLIS_CATEGORY_MAP skips ("Other substances", "Other",
    # "Steroids", "Analgesics"), so build_patterns() never compiles them. Only
    # 1,336 of the 3,101 NFLIS rows fall in mapped categories.
    #
    # That skip is right as a default -- mapping those categories in bulk pulls
    # ~150 routine chronic-disease medication mentions ("INSULIN DEPENDENT
    # DIABETES") and outright false positives ("EXPLOSION AT HEMP LABORATORY")
    # into the cohort. This is a per-term allowlist back out of the skip instead.
    #
    # HOW THESE TERMS WERE CHOSEN (and what is not known)
    # ---------------------------------------------------
    # Candidate pool: every substance name and synonym from the 1,765 skipped
    # NFLIS rows -- 1,891 unique terms. The filter that actually does the work
    # is corpus occurrence, not pharmacology: only 154 of those 1,891 appear
    # even once in the LA County ME corpus (2012-2026, 146,085 records), and
    # only 58 appear 5+ times. That is why this list is tens of terms and not
    # hundreds.
    #
    # What separated these terms from the other ~135 occurring candidates was
    # a by-hand read of their matches. That review is not reconstructable from
    # what is recorded here, so treat the list as reviewed judgment, not as
    # the output of a rule.
    #
    # In particular, do NOT reach for a "share of matches occurring in
    # toxicity language" score to justify or re-derive it. That was tried and
    # abandoned: the number is almost entirely an artifact of which phrases
    # the lexicon happens to include. Because this corpus writes drug deaths
    # as "EFFECTS OF <drug>" / "INTAKE OF DRUGS" rather than "<drug>
    # TOXICITY", loperamide and metoprolol score 50% and 72% under one word
    # list and 100% under another. Worse, the metric is scored per RECORD,
    # not per MENTION, so it cannot distinguish a poisoning from a therapy
    # mention sitting in the same record as an unrelated overdose -- and it
    # passes outright false positives such as the lithium record reading
    # "SEQUELAE OF thermal injuries ... lithium batteries when they exploded
    # and caught fire". It also does not discriminate: 22 of the 25
    # not-included candidates at n>=5 clear 90% on the corpus-matched
    # lexicon, the same as the terms kept here.
    #
    # The audit trail is reports/nflis_skipped_category_candidates.csv: all
    # 154 occurring candidates with corpus frequency, current status, and
    # rows each would add, so what was left out is inspectable even though
    # the reasoning is not recorded. Unresolved candidates scoring
    # indistinguishably from what is kept: ibuprofen (n=14), chlorcyclizine
    # (n=10), promethazine (n=20), baclofen (n=16), hydroxychloroquine
    # (n=14). dextromethorphan (n=6) is additionally blocked upstream by
    # "methorphan" in NFLIS_NAME_EXCLUSIONS, an exclusion that predates
    # clean_bert_text and may now be obsolete.
    #
    # False positives found by reading matches. "dapsone" and "amantadine"
    # were dropped from this list on that basis: dapsone's sole corpus match
    # is "NEUTROPENIA, POSSIBLY DAPSONE-INDUCED" (an adverse drug reaction,
    # not an overdose) and both amantadine matches are therapeutic
    # accumulation ("AMANTADINE ACCUMULATION | MEDICATION PRESCRIBED FOR").
    # Between them they contributed 3 rows, so removing them costs nothing.
    # One known FP remains, in "lithium": the battery fire quoted above, 1 of
    # 10 matches. It is kept because the other 9 are genuine lithium
    # toxicity; a narrower pattern excluding "lithium batter(y|ies)" would be
    # the fix if that 1 record matters.
    #
    # VALIDATION STATUS. Independent support comes from the external test set,
    # not the train set: 14 records across difluoroethane, acetaminophen,
    # ethylene glycol, loperamide and amlodipine are all labelled Others=1,
    # and none of those rows were touched by the 2026-07 label-correction
    # pass, so they are not circular. The train set gives no usable signal --
    # 10 of these terms never occur in it, and the 37 records where they do
    # occur with Others=0 are on inspection labelling gaps rather than false
    # positives (see reports/model_outputs/train_set_v2_others_allowlist_
    # review.csv), but those are proposed corrections, not yet applied.
    #
    # Counts below are rows each term newly adds to Others relative to
    # rosla_clean_pipe, on classified_all_deaths_07302026_regex.csv.
    (r"\bdifluoroethane\b",        "Others"),   # +127; inhalant ("huffing")
    (r"\bacetaminophen\b",         "Others"),   # +103; hepatotoxic OD
    (r"\bethylene\s+glycol\b",     "Others"),   # +23; antifreeze ingestion
    (r"\blithium\b",               "Others"),   # +7 (1 is a battery-fire FP, see above)
    (r"\bloperamide\b",            "Others"),   # +7
    (r"\bcolchicine\b",            "Others"),   # +4
    (r"\bmetaxalone\b",            "Others"),   # +1
    # Sodium/potassium nitrate self-poisoning. All 12 matches in this corpus are
    # ingestion cases, but note NFLIS lists isosorbide dinitrate/mononitrate --
    # the cardiac nitrates -- under "Other substances", so the bare-word pattern
    # is arguably too broad and behaves here only because therapy mentions are
    # phrased differently. The candidate table confirms the pattern does reach
    # "isosorbide mononitrate" (n=1) as well as "potassium nitrate" (n=1);
    # both are ingestion cases here, so it costs nothing in this corpus, but
    # the breadth is real. Reviewer's call.
    (r"\bnitrate[s]?\b",           "Others"),   # +12; 11 of 12 are SUICIDE mode
    # --- Cardiac/metabolic drugs. These are the class the wholesale-category
    # exclusion above exists to avoid, so they were reviewed match-by-match:
    # the occurrences here are ingestion/overdose contexts ("AMLODIPINE
    # TOXICITY", "METFORMIN INTOXICATION", "PROBABLE SEQUELAE OF CARVEDILOL AND
    # AMLODIPINE INTOXICATION"), not comorbidity mentions. Flagged for a second
    # opinion rather than assumed safe -- the case for them rests entirely on
    # that by-hand read, since no aggregate signal separates them from the
    # cardiac/metabolic candidates left out (atenolol, diltiazem,
    # propranolol). Note they skew heavily to intentional self-poisoning
    # rather than illicit use: of the rows they add, amlodipine is 12/17
    # SUICIDE mode and metoprolol 11/13, against 19% suicide for Others
    # overall. If the cohort is ever scoped to illicit/recreational overdose
    # specifically, this whole group needs revisiting.
    (r"\bamlodipine\b",            "Others"),   # +17
    (r"\bmetformin\b",             "Others"),   # +13
    (r"\bmetoprolol\b",            "Others"),   # +13
    (r"\bdigoxin\b",               "Others"),   # +4
    (r"\bfelodipine\b",            "Others"),   # +1
    (r"\bverapamil\b",             "Others"),   # +5
    (r"\btamsulosin\b",            "Others"),   # +1
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

# Generic "opioid(s)"/"opiate(s)" mention with no specific drug named: sets
# Any Opioids only, the same way GENERAL_DRUG_PATTERNS sets Any Drugs only.
# Validated against data/train_set_v2.csv (2026-07): "opioid"/"opioids" is a
# 100% consistent signal for Any Opioids=1 (81/81 records). "opiate"/
# "opiates" initially looked much less reliable (10/37 = 27%), but that
# turned out to be a train-set labeling gap, not a real difference in the
# term's reliability -- the 27 "opiate"-mentioning records with
# Any Opioids=0 were unambiguous opiate-toxicity/overdose cause-of-death
# text (e.g. "COMPLICATIONS OF OPIATE TOXICITY" with every other substance
# column also 0), not negated or ambiguous. Corrected in train_set_v2.csv;
# "opiate"/"opiates" is 100% precise against Any Opioids after the fix,
# same as "opioid"/"opioids".
GENERAL_OPIOID_PATTERNS = [
    r"\bopioids?\b",
    r"\bopiates?\b",
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
            if name.lower() in NFLIS_NAME_EXCLUSIONS:
                pass
            else:
                terms[NFLIS_NAME_OVERRIDE.get(name.lower(), col)].append(name)
        if pd.notna(row["Synonyms"]):
            for syn in str(row["Synonyms"]).split(";"):
                s = syn.strip()
                if s and s.lower() not in ("nan", "") and s.lower() not in NFLIS_NAME_EXCLUSIONS:
                    terms[NFLIS_NAME_OVERRIDE.get(s.lower(), col)].append(s)

    for _, row in nflis[nflis["NFLIS Detailed Drug Category"] == "Narcotics"].iterrows():
        col = NARCOTICS_OVERRIDE.get(str(row["Substance Name"]).strip().lower())
        if col:
            terms[col].append(str(row["Substance Name"]).strip())

    essential_by_col: dict[str, list[str]] = defaultdict(list)
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

    # Aggregate columns are driven by their own generic-phrase lists, not by
    # ESSENTIAL_PATTERNS, which only ever names specific substances.
    compiled["Any Drugs"] = re.compile("|".join(GENERAL_DRUG_PATTERNS), re.IGNORECASE)
    compiled["Any Opioids"] = re.compile("|".join(GENERAL_OPIOID_PATTERNS), re.IGNORECASE)
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


def clean_bert_text(value) -> str:
    """Normalize free text before it's tokenized for the multi-label BERT
    classifier. Single choke point for every quirk found in the 2026-07
    train/test formatting investigation:

      - literal "NULL" placeholder tokens (raw field concatenation used to
        build recoded_ext_test.csv leaves these in when a field is empty)
      - tab characters (used in recoded_ext_test.csv to join Primary.Cause
        and Secondary.Cause)
      - "comma after every single word" -- an artifact of whatever built
        combined_data_removing_mislabels.pkl's text column, NOT a real
        punctuation convention and NOT what create_text_col() below
        produces (which only joins CAUSE FIELDS with ", ", not individual
        words). Verified as a lossless ", " -> " " undo against samples.
        A model trained on this quirk scored 0.9985 F1 on in-distribution
        val data but only 0.7689 macro F1 on externally-formatted text --
        purely a formatting-robustness problem, confirmed by reformatting
        test text to match (0.9790) with zero retraining.
      - inconsistent casing -- train text is 100% uppercase; forcing
        uppercase here avoids reintroducing a train/inference case
        mismatch until a future retrain is done on case-diverse text.

    Idempotent: safe to call on text that's already been through this
    function, or through normalize_text(), or through create_text_col().
    """
    if pd.isna(value):
        return ""
    s = str(value)
    s = s.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\bNULL\b", " ", s, flags=re.IGNORECASE)
    s = s.replace(", ", " ")
    s = s.upper()
    return re.sub(r"\s+", " ", s).strip()


def _row_text(row: pd.Series) -> str:
    """Concatenate the cause-of-death text fields for a single record."""
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
    detail: dict[str, list] = {col: [] for col in SUBSTANCE_COLS + ["Any Drugs", "Any Opioids"]}
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
    # Generic opioid terms ("opioid", "opiate", …) set Any Opioids directly
    # without implying a specific subtype (Heroin/Fentanyl/Prescription.opioids).
    any_opioid_regex = regex_flags.get(
        "Any Opioids",
        np.zeros(len(out), dtype=int),
    )
    n_generic_opioid = int(any_opioid_regex.sum())

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
    original_any_opioids = (
        out["Any Opioids"].fillna(0).astype(int).values
        if "Any Opioids" in out.columns
        else np.zeros(len(out), dtype=int)
    )
    out["Any Opioids"] = (
        (subst[["Heroin", "Fentanyl", "Prescription.opioids"]].sum(axis=1) > 0).astype(int).values |
        original_any_opioids |
        any_opioid_regex
    ).astype(int)
    if verbose:
        before_opioid = int(original_any_opioids.sum())
        after_opioid = int(out["Any Opioids"].sum())
        sign = "+" if after_opioid >= before_opioid else ""
        print(
            f"  {'Any Opioids':<25} {before_opioid:>7,}  "
            f"{sign}{after_opioid - before_opioid:>5,}  {after_opioid:>7,}"
        )
        print(f"  Generic 'opioid(s)' mention matches: {n_generic_opioid:,}")

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
      - The cause-of-death text fields for manual inspection
      - All remaining original columns
    """
    compare_cols = SUBSTANCE_COLS + ["Any Drugs", "Any Opioids"]
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
    for col in SUBSTANCE_COLS + ["Any Drugs", "Any Opioids"]:
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