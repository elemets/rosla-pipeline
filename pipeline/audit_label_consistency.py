#!/usr/bin/env python
# coding: utf-8
"""
audit_label_consistency.py
===========================
Find label-noise candidates in a labeled dataset (validation set, internal
test set, train set, etc.) by checking whether records containing the same
unambiguous substance term are labeled consistently.

Method: for every column's compiled regex pattern (from
pipeline.regex_classifier.build_patterns), find every match in every
record's text, then group hits by the exact matched term. A term that
shows up with high precision (matches almost always co-occur with a
positive label) but has some negative-labeled matches is a strong
candidate for label noise on those specific records -- as opposed to a
term with low precision across the board, which usually reflects a
legitimate convention (e.g. generic "opioid"/"benzodiazepine" mentions
are deliberately not counted as positive unless a specific drug is named).

This script does NOT auto-correct anything. It produces six review
files for a human to look at before any labels are changed:

  1. <stem>_term_precision.csv     - precision of every (term, column) pair
  2. <stem>_label_review.csv       - candidate 0->1 corrections: individual
                                     records where a high-precision term
                                     matched but the label is still 0
  3. <stem>_vocab_gap.csv          - positive-labeled records where NO
                                     pattern matched at all. Usually a sign
                                     of missing vocabulary, but sometimes
                                     reveals the opposite kind of error (a
                                     positive label with no supporting
                                     evidence in the text at all -- e.g. we
                                     found "NONALCOHOLIC steatohepatitis"
                                     labeled Alcohol=1 in the train set this
                                     way, which is backwards: NASH is by
                                     definition NOT alcohol-related).
  4. <stem>_derived_consistency.csv - violations of the exact logical
                                     invariants that "Any Opioids" and
                                     "Any Drugs" are supposed to satisfy
                                     (e.g. Any Opioids must equal
                                     Heroin|Fentanyl|Prescription.opioids).
                                     Only written if those columns exist.
  5. <stem>_duplicate_conflicts.csv - groups of records sharing the exact
                                     same text but disagreeing on at least
                                     one label column. Direct evidence of
                                     inconsistency; no regex involved.
  6. <stem>_negation_review.csv    - label=1 records where every match of
                                     that column's pattern sits inside a
                                     negation ("denies alcohol use", "no
                                     history of opioid abuse"). Heuristic
                                     (word-window based) -- review by hand,
                                     this is not a hard rule like #4.

Usage (run from the repo root, as a module -- running it as a bare script
puts pipeline/ first on sys.path and collides with pipeline/pipeline.py):
    python -m pipeline.audit_label_consistency data/some_validation_set.csv \
        --text-col text --out-dir reports/model_outputs

Requires the CSV to have a text column and the 8 SUBSTANCE_COLS label
columns (Methamphetamine, Heroin, Cocaine, Fentanyl, Alcohol,
Prescription.opioids, Benzodiazepines, Others).

--apply mode
------------
Pass --apply to additionally auto-apply the subset of findings that don't
require human judgment, and write:

  <stem>_corrected.csv        - a full copy of the input CSV with those
                                 fixes applied (original file is never
                                 touched)
  <stem>_final_corrections.csv - a row-level log of every change, in the
                                 same schema as train_final_corrections.csv
                                 / external_test_final_corrections.csv
                                 (row_index, column, direction, matched_term,
                                 reason, text, source), source is "direct"
                                 for judgment-free label fixes or "derived"
                                 for the mechanical Any Opioids/Any Drugs
                                 recomputation that follows from them.

What gets auto-applied (deliberately conservative):
  - [2] label_review candidates, EXCEPT rows also flagged by [6] (every
    match negated) -- those are left in label_review.csv for manual review.
  - [4] derived_consistency rows with severity=error (exact invariant
    violations; the info-severity convention cases are never touched).
What does NOT get auto-applied, ever (left as review-only, same as without
--apply):
  - [3] vocab_gap: missing-vocabulary vs. wrong-label is inherently
    ambiguous from this signal alone.
  - [5] duplicate_conflicts: which of the conflicting labels is correct
    isn't decidable from the data itself.
--apply is a starting point, not a substitute for review: a term can be
unambiguous by this script's rules and still be a genuine judgment call
(e.g. whether Laennec's cirrhosis counts as alcohol-related was confirmed
by a human, not inferred).
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline.regex_classifier as rc

# Generic class-word terms with legitimately low precision -- these reflect
# a real labeling convention (only a *named* substance counts, not a
# generic class mention) and should not be treated as noise candidates.
# Confirmed against data/train_set.csv during the 2026-07 label audit.
CONVENTION_TERMS = {
    "opioid", "opioids", "opiate", "opiates",
    "benzodiazepine", "benzodiazepines",
}

# Columns "Any Opioids" is defined as the OR of (regex_classifier.py's
# apply_corrections, Stage: recompute derived columns).
OPIOID_COMPONENT_COLS = ["Heroin", "Fentanyl", "Prescription.opioids"]

# Heuristic negation triggers, checked within the word window preceding a
# match. Deliberately excludes bare "no"/"not"/"non" -- in death-certificate
# text those are almost always part of an unrelated word ("NONTRAUMATIC",
# "NO DEATH CERTIFICATE ISSUED", "COULD NOT BE DETERMINED") rather than a
# real negation of a substance finding (verified: 0/22,676 train_set_v2
# records contain a genuine multi-word negation phrase at all -- this
# corpus expresses absence by omission, not negation, so the bare triggers
# are pure false-positive risk here). The multi-word/specific phrases below
# are much less ambiguous and are what a narrative clinical-note dataset
# (e.g. a colleague's validation set) would actually use.
NEGATION_TRIGGERS = re.compile(
    r"\b("
    r"denies|denied|deny|without|"
    r"negative for|no evidence of|no evidence|no history of|no signs of|"
    r"ruled out|rule out|absence of"
    r")\b",
    re.IGNORECASE,
)
# Hard stop: never let a negation trigger reach across a SEARCH_FIELD
# boundary (regex_classifier._row_text joins CauseA..HowInjuryOccurred with
# " | "). Within a field we use a fixed trailing word-window rather than
# comma/period boundaries, since this corpus's text is comma-*tokenized*
# (one word per comma) rather than punctuated prose -- splitting on commas
# there would isolate every single word as its own "clause" and never see
# a preceding trigger word at all.
FIELD_BOUNDARY = re.compile(r"\|")
NEGATION_WINDOW_WORDS = 8
CONVENTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in CONVENTION_TERMS) + r")\b", re.IGNORECASE
)


def audit(
    csv_path: Path,
    text_col: str,
    out_dir: Path,
    precision_threshold: float,
    min_support: int,
    nflis_path: Path,
    apply: bool = False,
) -> None:
    rc.SEARCH_FIELDS = [text_col]
    df = pd.read_csv(csv_path)
    patterns = rc.build_patterns(nflis_path)
    texts = df[text_col].astype(str).tolist()

    missing_cols = [c for c in rc.SUBSTANCE_COLS if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Missing label columns in {csv_path}: {missing_cols}")

    # --- 1. Per-term precision -------------------------------------------
    stats: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    match_rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    labels = {col: df[col].astype(int).values for col in rc.SUBSTANCE_COLS}

    for col in rc.SUBSTANCE_COLS:
        pat = patterns[col]
        lab = labels[col]
        for i, text in enumerate(texts):
            for m in pat.findall(text):
                term = (m if isinstance(m, str) else str(m)).lower()
                key = (term, col)
                if lab[i] == 1:
                    stats[key][0] += 1
                else:
                    stats[key][1] += 1
                    match_rows[key].append(i)

    prec_rows = []
    for (term, col), (tp, fp) in stats.items():
        total = tp + fp
        prec_rows.append({
            "column": col, "term": term, "n_matches": total,
            "tp": tp, "fp": fp, "precision": tp / total if total else 0.0,
        })
    prec_df = pd.DataFrame(prec_rows).sort_values(["precision", "fp"], ascending=[True, False])

    stem = csv_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    prec_path = out_dir / f"{stem}_term_precision.csv"
    prec_df.to_csv(prec_path, index=False)

    # --- 2. Candidate 0->1 label corrections ------------------------------
    review_rows = []
    for _, r in prec_df.iterrows():
        term, col, precision, n = r["term"], r["column"], r["precision"], r["n_matches"]
        if term in CONVENTION_TERMS:
            continue
        if precision >= precision_threshold or n < min_support:
            continue
        for i in match_rows[(term, col)]:
            review_rows.append({
                "row_index": i, "column": col, "matched_term": term,
                "term_precision": round(precision, 3),
                "current_label": 0, "suggested_label": 1,
                "text": texts[i][:200],
            })
    review_df = pd.DataFrame(review_rows).drop_duplicates(["row_index", "column"])
    review_path = out_dir / f"{stem}_label_review.csv"
    review_df.to_csv(review_path, index=False)

    # --- 3. Vocabulary-gap / reverse-direction check ----------------------
    gap_rows = []
    for col in rc.SUBSTANCE_COLS:
        pat = patterns[col]
        lab = labels[col]
        for i, text in enumerate(texts):
            if lab[i] == 1 and not pat.search(text):
                gap_rows.append({"row_index": i, "column": col, "text": texts[i][:200]})
    gap_df = pd.DataFrame(gap_rows)
    gap_path = out_dir / f"{stem}_vocab_gap.csv"
    gap_df.to_csv(gap_path, index=False)

    # --- 4. Derived-column logical consistency (exact invariants) --------
    derived_rows = []
    have_opioid_cols = all(c in df.columns for c in OPIOID_COMPONENT_COLS + ["Any Opioids"])
    if have_opioid_cols:
        component_or = np.zeros(len(df), dtype=int)
        for c in OPIOID_COMPONENT_COLS:
            component_or = component_or | df[c].fillna(0).astype(int).values
        any_opioids = df["Any Opioids"].fillna(0).astype(int).values
        for i, (expected, actual) in enumerate(zip(component_or, any_opioids)):
            if expected != actual:
                # A generic "opioid(s)" mention with no specific drug named
                # legitimately sets Any Opioids=1 without setting a
                # component column -- same convention as CONVENTION_TERMS,
                # not a labeling bug.
                is_convention = actual == 1 and expected == 0 and bool(CONVENTION_PATTERN.search(texts[i]))
                derived_rows.append({
                    "row_index": i, "severity": "info" if is_convention else "error",
                    "check": (
                        "Any Opioids=1 from a generic 'opioid' mention with no named "
                        "substance (known convention)" if is_convention else
                        "Any Opioids != OR(Heroin, Fentanyl, Prescription.opioids)"
                    ),
                    "any_opioids": int(actual), "expected": int(expected),
                    "text": texts[i][:200],
                })
    have_any_drugs = "Any Drugs" in df.columns
    if have_any_drugs:
        subst_or = np.zeros(len(df), dtype=int)
        for c in rc.SUBSTANCE_COLS:
            subst_or = subst_or | df[c].fillna(0).astype(int).values
        any_drugs = df["Any Drugs"].fillna(0).astype(int).values
        for i in range(len(df)):
            if subst_or[i] == 1 and any_drugs[i] == 0:
                derived_rows.append({
                    "row_index": i, "severity": "error",
                    "check": "a substance column is 1 but Any Drugs is 0",
                    "any_opioids": "", "expected": "",
                    "text": texts[i][:200],
                })
            elif subst_or[i] == 0 and any_drugs[i] == 1:
                derived_rows.append({
                    "row_index": i, "severity": "info",
                    "check": "Any Drugs is 1 but no substance column is 1 "
                             "(may be a legitimate generic drug-death phrase)",
                    "any_opioids": "", "expected": "",
                    "text": texts[i][:200],
                })
    derived_df = pd.DataFrame(derived_rows)
    derived_path = out_dir / f"{stem}_derived_consistency.csv"
    derived_df.to_csv(derived_path, index=False)

    # --- 5. Duplicate-text label conflicts --------------------------------
    dup_check_cols = [c for c in rc.SUBSTANCE_COLS + ["Any Opioids", "Any Drugs"] if c in df.columns]
    dup_rows = []
    group_id = 0
    for text_val, group in df[df[text_col].astype(str).str.strip() != ""].groupby(df[text_col].astype(str)):
        if len(group) < 2:
            continue
        conflict_cols = [c for c in dup_check_cols if group[c].astype(int).nunique() > 1]
        if not conflict_cols:
            continue
        group_id += 1
        for c in conflict_cols:
            dup_rows.append({
                "dup_group_id": group_id,
                "column": c,
                "n_rows": len(group),
                "row_indices": ",".join(str(i) for i in group.index),
                "values": ",".join(str(v) for v in group[c].astype(int).tolist()),
                "text": text_val[:200],
            })
    dup_df = pd.DataFrame(dup_rows)
    dup_path = out_dir / f"{stem}_duplicate_conflicts.csv"
    dup_df.to_csv(dup_path, index=False)

    # --- 6. Negation-scope review ------------------------------------------
    neg_rows = []
    for col in rc.SUBSTANCE_COLS:
        pat = patterns[col]
        lab = labels[col]
        for i, text in enumerate(texts):
            if lab[i] != 1:
                continue
            matches = list(pat.finditer(text))
            if not matches:
                continue
            negated_terms, any_unnegated = [], False
            for m in matches:
                field_start = 0
                for b in FIELD_BOUNDARY.finditer(text, 0, m.start()):
                    field_start = b.end()
                preceding_words = text[field_start:m.start()].replace(",", " ").split()
                window = " ".join(preceding_words[-NEGATION_WINDOW_WORDS:])
                if NEGATION_TRIGGERS.search(window):
                    negated_terms.append(m.group())
                else:
                    any_unnegated = True
            if negated_terms and not any_unnegated:
                neg_rows.append({
                    "row_index": i, "column": col,
                    "negated_terms": ", ".join(sorted(set(negated_terms))),
                    "text": text[:200],
                })
    neg_df = pd.DataFrame(neg_rows)
    neg_path = out_dir / f"{stem}_negation_review.csv"
    neg_df.to_csv(neg_path, index=False)

    # --- apply: auto-apply the judgment-free subset of findings -----------
    corrected_path = final_corrections_path = None
    n_applied_direct = n_applied_derived = n_skipped_negated = 0
    if apply:
        out = df.copy()
        applied_rows = []
        negated_keys = {(r["row_index"], r["column"]) for r in neg_rows}
        substance_changed = set()  # row_index -> True if a substance col changed this row

        for _, r in review_df.iterrows():
            key = (r["row_index"], r["column"])
            if key in negated_keys:
                n_skipped_negated += 1
                continue
            i, col = r["row_index"], r["column"]
            out.loc[i, col] = 1
            substance_changed.add(i)
            applied_rows.append({
                "row_index": i, "column": col, "direction": "0_to_1",
                "matched_term": r["matched_term"],
                "reason": f"term precision {r['term_precision']} below threshold={precision_threshold}, not a known convention term",
                "text": texts[i][:200], "source": "direct",
            })
            n_applied_direct += 1

        # Recompute Any Drugs: OR of substance cols, never flip 1->0 (the
        # "1 with no substance col" case is only ever info, never error).
        if have_any_drugs:
            subst_or_new = np.zeros(len(out), dtype=int)
            for c in rc.SUBSTANCE_COLS:
                subst_or_new = subst_or_new | out[c].fillna(0).astype(int).values
            old_any_drugs = df["Any Drugs"].fillna(0).astype(int).values
            new_any_drugs = old_any_drugs | subst_or_new
            for i in range(len(out)):
                if new_any_drugs[i] != old_any_drugs[i]:
                    out.loc[i, "Any Drugs"] = int(new_any_drugs[i])
                    reason = (
                        "mechanical recomputation from a substance-column change in the same row"
                        if i in substance_changed else
                        "a substance column is 1 but Any Drugs was 0 (derived-consistency error fix)"
                    )
                    applied_rows.append({
                        "row_index": i, "column": "Any Drugs", "direction": "0_to_1",
                        "matched_term": "", "reason": reason,
                        "text": texts[i][:200], "source": "derived",
                    })
                    n_applied_derived += 1

        # Recompute Any Opioids: OR of components, EXCEPT preserve a
        # convention-driven 1 (generic "opioid" word, no named component).
        if have_opioid_cols:
            component_or_new = np.zeros(len(out), dtype=int)
            for c in OPIOID_COMPONENT_COLS:
                component_or_new = component_or_new | out[c].fillna(0).astype(int).values
            old_any_opioids = df["Any Opioids"].fillna(0).astype(int).values
            for i in range(len(out)):
                if component_or_new[i] == 1:
                    new_val = 1
                else:
                    is_convention = old_any_opioids[i] == 1 and bool(CONVENTION_PATTERN.search(texts[i]))
                    new_val = 1 if is_convention else 0
                if new_val != old_any_opioids[i]:
                    out.loc[i, "Any Opioids"] = new_val
                    direction = "0_to_1" if new_val == 1 else "1_to_0"
                    reason = (
                        "mechanical recomputation from an opioid-component column change in the same row"
                        if i in substance_changed else
                        "Any Opioids invariant violated with no supporting component or convention word (derived-consistency error fix)"
                    )
                    applied_rows.append({
                        "row_index": i, "column": "Any Opioids", "direction": direction,
                        "matched_term": "", "reason": reason,
                        "text": texts[i][:200], "source": "derived",
                    })
                    n_applied_derived += 1

        corrected_path = out_dir / f"{stem}_corrected.csv"
        out.to_csv(corrected_path, index=False)

        final_df = pd.DataFrame(applied_rows).sort_values(["row_index", "column"]).reset_index(drop=True) \
            if applied_rows else pd.DataFrame(columns=["row_index", "column", "direction", "matched_term", "reason", "text", "source"])
        final_corrections_path = out_dir / f"{stem}_final_corrections.csv"
        final_df.to_csv(final_corrections_path, index=False)

    print(f"Source: {csv_path} ({len(df)} records)")
    print(f"\n[1] Term precision table: {len(prec_df)} distinct (term, column) pairs -> {prec_path}")
    low = prec_df[(prec_df["precision"] < precision_threshold) & (prec_df["n_matches"] >= min_support)]
    print(f"    {len(low)} pairs below precision={precision_threshold} with n>={min_support} (excluding {len(CONVENTION_TERMS)} known convention terms)")
    print(f"\n[2] Candidate 0->1 label corrections: {len(review_df)} records -> {review_path}")
    print(f"\n[3] Positive-labeled records with zero pattern match: {len(gap_df)} -> {gap_path}")
    if len(gap_df):
        print("    Review these by hand: usually missing vocabulary, but check for the")
        print("    'reverse' case too (positive label with no real evidence in the text).")
    if have_opioid_cols or have_any_drugs:
        n_err = int((derived_df["severity"] == "error").sum()) if len(derived_df) else 0
        n_info = int((derived_df["severity"] == "info").sum()) if len(derived_df) else 0
        print(f"\n[4] Derived-column consistency: {n_err} hard violations, {n_info} informational -> {derived_path}")
    else:
        print("\n[4] Derived-column consistency: skipped (Any Opioids / Any Drugs columns not present)")
    print(f"\n[5] Duplicate-text label conflicts: {group_id} conflicting groups -> {dup_path}")
    print(f"\n[6] Negation-scope review: {len(neg_df)} records where every match is negated -> {neg_path}")
    if apply:
        print(f"\n[apply] {n_applied_direct} direct + {n_applied_derived} derived corrections applied "
              f"({n_skipped_negated} label_review candidates skipped, negated) ->")
        print(f"    {corrected_path}")
        print(f"    {final_corrections_path}")
        print("    vocab_gap and duplicate_conflicts were NOT auto-applied -- review those by hand.")
    else:
        print("\nNothing has been changed. Review the CSVs above, then apply corrections")
        print("by hand the same way data/recoded_ext_test_v2.csv and data/train_set_v2.csv were built")
        print("(or re-run with --apply to auto-apply the judgment-free subset).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", type=Path, help="Labeled CSV to audit (validation set, internal test set, etc.)")
    parser.add_argument("--text-col", default="text", help="Name of the free-text column (default: text)")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/model_outputs"), help="Where to write the review CSVs")
    parser.add_argument("--precision-threshold", type=float, default=0.90, help="Flag terms below this precision (default: 0.90)")
    parser.add_argument("--min-support", type=int, default=1, help="Ignore terms with fewer than this many total matches (default: 1 -- low-support hits are still surfaced, but review them by hand since n=1-2 isn't a statistically reliable signal)")
    parser.add_argument("--nflis-path", type=Path, default=rc.DEFAULT_NFLIS, help="NFLIS substances CSV used to build patterns")
    parser.add_argument("--apply", action="store_true", help="Auto-apply the judgment-free subset of findings; writes <stem>_corrected.csv and <stem>_final_corrections.csv (see --help epilog)")
    args = parser.parse_args()

    audit(
        csv_path=args.csv_path,
        text_col=args.text_col,
        out_dir=args.out_dir,
        precision_threshold=args.precision_threshold,
        min_support=args.min_support,
        nflis_path=args.nflis_path,
        apply=args.apply,
    )


if __name__ == "__main__":
    main()
