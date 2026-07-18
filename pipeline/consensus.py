"""
Consensus collation of raw coroner files.

Collapses duplicate CaseNumbers across all raw source files by majority vote
per column: the value most files agree on wins, ties are broken by each
file's overall reliability (how often that file agrees with the consensus
elsewhere). The result is one maximally complete row per case built from
values that are in agreement across files, which limits the damage a single
corrupted extraction can do.

Also produces a per-file audit reporting how often each file disagrees with
the cross-file consensus, to flag corrupted extractions.

Usage:
    python consensus.py [raw_dir] [--out consensus.csv] [--report report.csv]
"""

import argparse
import glob
import os
import re

import pandas as pd

COL_MAP = {
    "CaseNumber": ["CaseNumber", "CaseNum", "Case Number", "Case#"],
    "ResidenceType": ["ResidenceType", "ResType", "Residence Type"],
    "DeathDate": ["DeathDate", "Date of Death", "Death Date", "DateOfDeath", "DateofDeath"],
    "DeathTime": ["DeathTime", "Time of Death", "Death Time", "TimeofDeath"],
    "DeathPlace": ["DeathPlace", "Death Place"],
    "DeathAddress": ["DeathAddress", "DeathAddr", "DeathAdress", "DeathAddr.1", "address.death", "DeathAdress.1", "Death Address"],
    "DeathZip": ["DeathZip", "DeathZip.1", "DeathZipCode", "DeathZi\np", "Zip"],
    "DeathCity": ["DeathCity", "Death City", "DeathCityDesc"],
    "EventPlace": ["EventPlace", "Event Place"],
    "EventAddress": ["EventAddress", "EventAddr", "EventAddr.1", "eventaddress", "Event Address"],
    "EventZip": ["EventZip", "EventZip.1", "EventZi\np", "EventZipCode", "Zip.1", "Event Zip"],
    "EventCity": ["EventCity", "EventCityDesc"],
    "Mode": ["Mode", "Mode.1"],
    "CauseA": ["CauseA", "Cause A", "DeathCauseA"],
    "CauseB": ["CauseB", "Cause B", "DeathCauseB"],
    "CauseC": ["CauseC", "Cause C", "DeathCauseC"],
    "CauseD": ["CauseD", "Cause D", "DeathCauseD"],
    "CauseOther": ["CauseOther", "Other Cause", "OtherCause"],
    "HowInjuryOccurred": ["HowInjuryOccurred", "InjuryDesc", "HowInjuryOccu\nrred"],
    "FirstName": ["FirstName", "First Name"],
    "MiddleName": ["MiddleName", "Middle Name"],
    "LastName": ["LastName", "Last Name"],
    "DateofBirth": ["DateofBirth", "Date of Birth", "BirthDate"],
    "Race": ["Race", "Races"],
    "Gender": ["Gender", "Sex"],
    "Age": ["Age"],
    "SourcePages": ["SourcePages"],
}

# Columns that legitimately differ between extractions (page numbers belong
# to each source PDF), so they never count for or against agreement.
NO_VOTE_COLS = {"SourcePages"}

NULL_TOKENS = {"", "NULL", "NAN", "NONE", "N/A", "NA", "UNKNOWN", "UNKNOWNOTHER", "OTHER"}

CASE_RE = re.compile(r"^\d{4}-\d+$")

# Canonical race buckets keyed by substring of the alphanumeric-only value.
RACE_BUCKETS = [
    ("HISPANIC", "LATINE"),
    ("LATIN", "LATINE"),
    ("CAUCASIAN", "WHITE"),
    ("WHITE", "WHITE"),
    ("BLACK", "BLACK"),
    ("AFRICANAMERICAN", "BLACK"),
    ("AMERICANINDIAN", "AMERICAN INDIAN"),
    ("NATIVEAMERICAN", "AMERICAN INDIAN"),
    ("PACIFICISLANDER", "PACIFIC ISLANDER"),
    ("ASIAN", "ASIAN"),
    ("MIDDLEEASTERN", "MIDDLE EASTERN"),
]

# A value shorter than this never merges into a longer one as a truncation.
MIN_TRUNCATION_LEN = 5


def load_raw_files(raw_dir):
    """Load every csv/xlsx in raw_dir into one canonical-column DataFrame."""
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")) + glob.glob(os.path.join(raw_dir, "*.xlsx")))
    frames = []
    for path in paths:
        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path, dtype=str, low_memory=False)
            else:
                df = pd.read_excel(path, dtype=str)
        except Exception as e:
            print(f"skipping {path}: {e}")
            continue
        out = pd.DataFrame(index=df.index)
        for canon, aliases in COL_MAP.items():
            for alias in aliases:
                if alias in df.columns:
                    out[canon] = df[alias]
                    break
        if "CaseNumber" not in out.columns:
            print(f"skipping {path}: no case number column")
            continue
        out["CaseNumber"] = out["CaseNumber"].astype(str).str.strip()
        out = out[out["CaseNumber"].str.match(CASE_RE)]
        out["source_file"] = os.path.basename(path)
        frames.append(out)
        print(f"loaded {os.path.basename(path)}: {len(out)} rows")
    return pd.concat(frames, ignore_index=True)


def normalize_series(s, column):
    """
    Canonicalize values of one column into comparison keys.

    Keys are uppercase and stripped of all non-alphanumeric characters, so
    linebreaks inside words ("METHAMPHETAMI\\nNE"), stray spaces and
    punctuation variants all collapse to the same key.
    """
    s = s.astype(str).str.upper()
    if column in ("DeathDate", "DateofBirth"):
        return pd.to_datetime(s, errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
    if column == "Age":
        return s.str.extract(r"(\d+)", expand=False)
    if column in ("DeathZip", "EventZip"):
        return s.str.replace(r"\.0$", "", regex=True).str.extract(r"(\d{5})", expand=False)
    # Drop null-ish word tokens first: "NULL NULL", "NULL FALSE" and
    # "DEFERRED" causes are all absence of information, not values.
    s = s.str.replace(r"\b(NULL|NONE|NAN|N/?A|TRUE|FALSE|UNKNOWN|PENDING|DEFERRED)\b", " ", regex=True)
    s = s.str.replace(r"[^A-Z0-9]", "", regex=True)
    s = s.mask(s.isin(NULL_TOKENS) | (s == ""))
    if column == "Gender":
        return s.replace({"F": "FEMALE", "M": "MALE"})
    if column == "Race":
        keys = pd.Series(pd.NA, index=s.index, dtype=object)
        for substring, bucket in RACE_BUCKETS:
            keys = keys.mask(keys.isna() & s.str.contains(substring, na=False), bucket)
        return keys.where(keys.notna(), s)
    return s


def build_long(df):
    """Melt to one row per (case, column, source) non-null cell."""
    value_cols = [c for c in df.columns if c not in ("CaseNumber", "source_file")]
    long = df.melt(
        id_vars=["CaseNumber", "source_file"],
        value_vars=value_cols,
        var_name="column",
        value_name="value",
    ).dropna(subset=["value"])
    norms = []
    for col, grp in long.groupby("column", sort=False):
        norms.append(normalize_series(grp["value"], col))
    long["norm"] = pd.concat(norms)
    long = long.dropna(subset=["norm"])
    # One vote per file per cell: a file repeating a case shouldn't count twice.
    long = long.drop_duplicates(subset=["CaseNumber", "column", "source_file", "norm"])
    return long


def merge_truncated_votes(votes):
    """
    Fold truncated values into their full versions before counting.

    Extractions frequently cut cells short ("MULTIPLE GUNSHOT" vs "MULTIPLE
    GUNSHOT WOUNDS"). Within each (case, column) group, a value whose key is
    a prefix of a longer value's key transfers its votes to the longer value
    instead of competing with it.
    """
    contested_mask = votes.duplicated(["CaseNumber", "column"], keep=False)
    settled = votes[~contested_mask]
    merged_groups = []
    for _, group in votes[contested_mask].groupby(["CaseNumber", "column"], sort=False):
        rows = group.sort_values("norm", key=lambda s: s.str.len(), ascending=False).to_dict("records")
        kept = []
        for row in rows:
            target = next(
                (k for k in kept
                 if len(row["norm"]) >= MIN_TRUNCATION_LEN
                 and (k["norm"].startswith(row["norm"]) or k["norm"].endswith(row["norm"]))),
                None,
            )
            if target is not None:
                target["count"] += row["count"]
            else:
                kept.append(row)
        merged_groups.extend(kept)
    merged = pd.DataFrame(merged_groups, columns=votes.columns)
    return pd.concat([settled, merged], ignore_index=True)


def vote(long):
    """
    Two-pass majority vote.

    Pass 1 picks a provisional winner per (case, column) by raw vote count and
    scores each file's reliability as its share of contested cells where it
    voted with the plurality. Pass 2 re-runs the vote with reliability as the
    tie-breaker, so a corrupted file loses 1-vs-1 conflicts against a file
    that agrees with the consensus everywhere else.
    """
    # Representative original value per key: the one with the fewest
    # whitespace breaks, so "Hispanic/Latino" beats "Hispanic/L\natino".
    long = long.assign(_breaks=long["value"].str.count(r"\s")).sort_values(
        "_breaks", kind="stable"
    )
    votes = (
        long.groupby(["CaseNumber", "column", "norm"], sort=False)
        .agg(count=("source_file", "size"), value=("value", "first"))
        .reset_index()
    )
    long = long.drop(columns="_breaks")
    votes = merge_truncated_votes(votes)

    provisional = votes.sort_values("count", ascending=False, kind="stable").drop_duplicates(
        ["CaseNumber", "column"]
    )
    contested_keys = votes.duplicated(["CaseNumber", "column"], keep=False)
    contested = votes[contested_keys][["CaseNumber", "column"]].drop_duplicates()

    scored = long.merge(contested, on=["CaseNumber", "column"]).merge(
        provisional[["CaseNumber", "column", "norm"]].rename(columns={"norm": "winner"}),
        on=["CaseNumber", "column"],
    )
    scored = scored[~scored["column"].isin(NO_VOTE_COLS)]
    # A truncated fragment of the winning value counts as agreement.
    scored["agree"] = [
        n == w or (len(n) >= MIN_TRUNCATION_LEN and (w.startswith(n) or w.endswith(n)))
        for n, w in zip(scored["norm"], scored["winner"])
    ]
    reliability = scored.groupby("source_file")["agree"].mean().rename("reliability")

    backer_rel = (
        long.merge(reliability, on="source_file", how="left")
        .groupby(["CaseNumber", "column", "norm"], sort=False)["reliability"]
        .max()
        .reset_index()
    )
    votes = votes.merge(backer_rel, on=["CaseNumber", "column", "norm"], how="left")
    winners = votes.sort_values(
        ["count", "reliability"], ascending=False, kind="stable"
    ).drop_duplicates(["CaseNumber", "column"])

    return winners, reliability, scored


def consensus_collate(raw_dir):
    """Return (consensus_df, file_report, cell_disagreements) for raw_dir."""
    df = load_raw_files(raw_dir)
    long = build_long(df)
    winners, reliability, scored = vote(long)

    consensus = winners.pivot(index="CaseNumber", columns="column", values="value")
    ordered = [c for c in COL_MAP if c in consensus.columns and c != "CaseNumber"]
    consensus = consensus[ordered].reset_index()
    # PDF extractions leave linebreaks inside values; flatten for output.
    consensus = consensus.replace(r"\s+", " ", regex=True)

    per_file = (
        scored.groupby("source_file")
        .agg(
            contested_cells=("agree", "size"),
            disagreements=("agree", lambda a: int((~a).sum())),
        )
        .reset_index()
    )
    total_cells = (
        long[~long["column"].isin(NO_VOTE_COLS)]
        .groupby("source_file")
        .size()
        .rename("total_cells")
    )
    rows_per_file = df.groupby("source_file").size().rename("rows")
    report = (
        per_file.merge(rows_per_file, on="source_file")
        .merge(total_cells, on="source_file")
        .merge(reliability, on="source_file")
    )
    report["disagreement_rate"] = report["disagreements"] / report["contested_cells"]
    report = report.sort_values("disagreement_rate", ascending=False)

    losers = (
        scored[~scored["agree"]][["CaseNumber", "column", "source_file", "value"]]
        .rename(columns={"value": "file_value"})
        .merge(
            winners[["CaseNumber", "column", "value"]].rename(columns={"value": "consensus_value"}),
            on=["CaseNumber", "column"],
            how="left",
        )
    )
    losers[["file_value", "consensus_value"]] = losers[["file_value", "consensus_value"]].replace(
        r"\s+", " ", regex=True
    )

    return consensus, report, losers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", nargs="?", default="pipeline_steps/input_files/raw")
    parser.add_argument("--out", default="consensus_all_deaths.csv")
    parser.add_argument("--report", default="consensus_file_report.csv")
    parser.add_argument("--disagreements", default="consensus_disagreements.csv")
    args = parser.parse_args()

    consensus, report, losers = consensus_collate(args.raw_dir)

    consensus.to_csv(args.out, index=False)
    report.to_csv(args.report, index=False)
    losers.to_csv(args.disagreements, index=False)

    print(f"\nconsensus rows: {len(consensus)} -> {args.out}")
    print(f"cell-level disagreements: {len(losers)} -> {args.disagreements}")
    print(f"\nfile agreement report ({args.report}):")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
