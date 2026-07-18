import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_COMPARE_COLUMNS = [
    "DeathDate",
    "FirstName",
    "MiddleName",
    "LastName",
    "DateofBirth",
    "Age",
    "Gender",
    "Race",
    "Races",
    "Mode",
    "CauseA",
    "CauseB",
    "CauseC",
    "CauseD",
    "CauseOther",
    "HowInjuryOccurred",
    "Methamphetamine",
    "Heroin",
    "Cocaine",
    "Fentanyl",
    "Alcohol",
    "Prescription.opioids",
    "Any Opioids",
    "Benzodiazepines",
    "Others",
    "Any Drugs",
    "Number_Substances",
    "Polysubstance",
    "DeathPlace",
    "DeathAddress",
    "DeathCity",
    "DeathZip",
    "EventPlace",
    "EventAddress",
    "EventCity",
    "EventZip",
    "ResidenceType",
    "ExperiencingHomelessness",
    "lon",
    "lat",
    "in_la_county",
]


def parse_dates(series):
    return pd.to_datetime(series, errors="coerce", format="mixed")


def normalize_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def load_year(path, year, key_column):
    df = pd.read_csv(path, low_memory=False)
    return filter_year(df, path, year, key_column)


def filter_year(df, label, year, key_column):
    if key_column not in df.columns:
        raise ValueError(f"{label} does not have key column '{key_column}'")
    if "DeathDate" not in df.columns:
        raise ValueError(f"{label} does not have a DeathDate column")

    df = df.copy()
    df["_parsed_death_date"] = parse_dates(df["DeathDate"])
    df = df[df["_parsed_death_date"].dt.year == year].copy()
    df = df[df[key_column].notna()].copy()

    # Keep the last occurrence for a case within a snapshot. The pipeline's
    # master output should already be one row per CaseNumber, but older
    # snapshots may not be perfectly deduplicated.
    df["_row_order"] = range(len(df))
    df = df.sort_values("_row_order").drop_duplicates(key_column, keep="last")
    df[key_column] = df[key_column].map(normalize_value)
    return df.set_index(key_column, drop=False)


def coalesce_duplicate_rows(df, subset, tie_keep="last"):
    if df.empty or subset not in df.columns:
        return df

    original_columns = df.columns.tolist()
    completeness = df.notna().sum(axis=1)
    df_ranked = df.assign(_completeness=completeness)
    if tie_keep == "last":
        df_ranked = df_ranked[::-1]

    df_sorted = (
        df_ranked.sort_values("_completeness", ascending=False, kind="stable")
        .drop(columns="_completeness")
    )
    combined = df_sorted.groupby(
        subset, as_index=False, sort=False, dropna=False
    ).first()
    return combined[original_columns]


def comparable_columns(old_df, new_df, key_column, extra_columns=None):
    columns = list(DEFAULT_COMPARE_COLUMNS)
    if extra_columns:
        columns.extend(extra_columns)

    present = []
    for column in columns:
        if column == key_column:
            continue
        if column in old_df.columns and column in new_df.columns and column not in present:
            present.append(column)
    return present


def compare_snapshots(old_path, new_path, year=2025, key_column="CaseNumber", extra_columns=None):
    old_df = load_year(old_path, year, key_column)
    new_df = load_year(new_path, year, key_column)
    return compare_year_frames(
        old_df=old_df,
        new_df=new_df,
        old_label=str(old_path),
        new_label=str(new_path),
        year=year,
        key_column=key_column,
        extra_columns=extra_columns,
    )


def compare_year_frames(
    old_df,
    new_df,
    old_label,
    new_label,
    year=2025,
    key_column="CaseNumber",
    extra_columns=None,
):

    old_keys = set(old_df.index)
    new_keys = set(new_df.index)
    added_keys = sorted(new_keys - old_keys)
    removed_keys = sorted(old_keys - new_keys)
    shared_keys = sorted(old_keys & new_keys)
    columns = comparable_columns(old_df, new_df, key_column, extra_columns)

    changed_rows = []
    for key in shared_keys:
        for column in columns:
            old_value = normalize_value(old_df.at[key, column])
            new_value = normalize_value(new_df.at[key, column])
            if old_value != new_value:
                changed_rows.append(
                    {
                        key_column: key,
                        "column": column,
                        "old_value": old_value,
                        "new_value": new_value,
                    }
                )

    changed_df = pd.DataFrame(changed_rows)
    changed_case_count = changed_df[key_column].nunique() if not changed_df.empty else 0

    summary = pd.DataFrame(
        [
            {
                "year": year,
                "old_file": old_label,
                "new_file": new_label,
                "old_cases": len(old_df),
                "new_cases": len(new_df),
                "net_case_delta": len(new_df) - len(old_df),
                "added_cases": len(added_keys),
                "removed_cases": len(removed_keys),
                "changed_existing_cases": changed_case_count,
                "changed_cells": len(changed_df),
                "unchanged_existing_cases": len(shared_keys) - changed_case_count,
            }
        ]
    )

    added_df = new_df.loc[added_keys].reset_index(drop=True) if added_keys else pd.DataFrame()
    removed_df = old_df.loc[removed_keys].reset_index(drop=True) if removed_keys else pd.DataFrame()

    old_monthly = old_df["_parsed_death_date"].dt.to_period("M").value_counts().sort_index()
    new_monthly = new_df["_parsed_death_date"].dt.to_period("M").value_counts().sort_index()
    monthly = (
        pd.DataFrame({"old_cases": old_monthly, "new_cases": new_monthly})
        .fillna(0)
        .astype(int)
        .reset_index(names="month")
    )
    monthly["month"] = monthly["month"].astype(str)
    monthly["net_case_delta"] = monthly["new_cases"] - monthly["old_cases"]

    if "source_file" in added_df.columns and not added_df.empty:
        added_by_source = (
            added_df.groupby("source_file", dropna=False)
            .size()
            .reset_index(name="added_cases")
            .sort_values("added_cases", ascending=False)
        )
    else:
        added_by_source = pd.DataFrame(columns=["source_file", "added_cases"])

    return {
        "summary": summary,
        "monthly": monthly,
        "added_cases": added_df,
        "removed_cases": removed_df,
        "changed_cells": changed_df,
        "added_by_source": added_by_source,
    }


def geocoded_name_for_raw(raw_file):
    return f"{Path(raw_file).stem}_geocoded.csv"


def ordered_geocoded_files(geocoded_dir, processed_log=None, order="processed_log"):
    geocoded_path = Path(geocoded_dir)
    available = {
        path.name: path
        for path in geocoded_path.glob("*_geocoded.csv")
        if path.name != "combined_classified_data_geocoded.csv"
    }

    if order == "processed_log" and processed_log:
        ordered = []
        seen = set()
        with open(processed_log) as f:
            for line in f:
                raw_file = line.strip()
                if not raw_file:
                    continue
                name = geocoded_name_for_raw(raw_file)
                if name in available:
                    ordered.append(available[name])
                    seen.add(name)

        ordered.extend(
            sorted(
                (path for name, path in available.items() if name not in seen),
                key=lambda path: path.name,
            )
        )
        return ordered

    if order == "mtime":
        return sorted(available.values(), key=lambda path: path.stat().st_mtime)

    return sorted(available.values(), key=lambda path: path.name)


def reconstruct_stability(
    geocoded_dir,
    year=2025,
    key_column="CaseNumber",
    processed_log=None,
    order="processed_log",
    extra_columns=None,
):
    files = ordered_geocoded_files(
        geocoded_dir=geocoded_dir,
        processed_log=processed_log,
        order=order,
    )
    if not files:
        raise ValueError(f"No per-file *_geocoded.csv files found in {geocoded_dir}")

    combined = pd.DataFrame()
    timeline_rows = []
    all_added = []
    all_removed = []
    all_changed = []
    all_monthly = []

    for step, path in enumerate(files, start=1):
        new_file_df = pd.read_csv(path, low_memory=False)
        before_combined = combined.copy()
        before_year = (
            filter_year(before_combined, f"before step {step}", year, key_column)
            if not before_combined.empty
            else pd.DataFrame()
        )

        combined = pd.concat([combined, new_file_df], ignore_index=True)
        combined = coalesce_duplicate_rows(combined, subset=key_column, tie_keep="last")
        after_year = filter_year(combined, f"after step {step}", year, key_column)

        if before_year.empty:
            before_year = after_year.iloc[0:0].copy()

        comparison = compare_year_frames(
            old_df=before_year,
            new_df=after_year,
            old_label=f"before {path.name}",
            new_label=f"after {path.name}",
            year=year,
            key_column=key_column,
            extra_columns=extra_columns,
        )

        source_year = filter_year(new_file_df, path.name, year, key_column)
        summary = comparison["summary"].iloc[0].to_dict()
        summary.update(
            {
                "step": step,
                "added_file": path.name,
                "file_rows_in_year": len(source_year),
                "file_unique_cases_in_year": source_year.index.nunique(),
            }
        )
        timeline_rows.append(summary)

        for output_name, collector in [
            ("added_cases", all_added),
            ("removed_cases", all_removed),
            ("changed_cells", all_changed),
            ("monthly", all_monthly),
        ]:
            df = comparison[output_name].copy()
            if not df.empty:
                df.insert(0, "added_file", path.name)
                df.insert(0, "step", step)
                collector.append(df)

    timeline = pd.DataFrame(timeline_rows)
    return {
        "timeline": timeline,
        "monthly_by_step": pd.concat(all_monthly, ignore_index=True)
        if all_monthly
        else pd.DataFrame(),
        "added_cases_by_step": pd.concat(all_added, ignore_index=True)
        if all_added
        else pd.DataFrame(),
        "removed_cases_by_step": pd.concat(all_removed, ignore_index=True)
        if all_removed
        else pd.DataFrame(),
        "changed_cells_by_step": pd.concat(all_changed, ignore_index=True)
        if all_changed
        else pd.DataFrame(),
    }


def write_outputs(results, output_dir, prefix):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written = {}
    for name, df in results.items():
        path = output_path / f"{prefix}_{name}.csv"
        df.to_csv(path, index=False)
        written[name] = path
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Compare two pipeline master outputs to assess year-specific stability."
    )
    parser.add_argument("--old", help="Older master/geocoded CSV.")
    parser.add_argument("--new", help="Newer master/geocoded CSV.")
    parser.add_argument(
        "--reconstruct-dir",
        help="Directory of per-file *_geocoded.csv files to replay one at a time.",
    )
    parser.add_argument(
        "--processed-log",
        default="pipeline_steps/logs/processed_files.txt",
        help="processed_files.txt used to order reconstructed file additions.",
    )
    parser.add_argument(
        "--order",
        choices=["processed_log", "mtime", "name"],
        default="processed_log",
        help="Order for --reconstruct-dir replay.",
    )
    parser.add_argument("--year", type=int, default=2025, help="DeathDate year to audit.")
    parser.add_argument("--key", default="CaseNumber", help="Case identifier column.")
    parser.add_argument(
        "--outdir",
        default="pipeline_steps/audits",
        help="Directory where audit CSVs should be written.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to stability_<year>_<timestamp>.",
    )
    parser.add_argument(
        "--compare-column",
        action="append",
        default=[],
        help="Additional column to compare. Can be passed more than once.",
    )
    args = parser.parse_args()

    prefix = args.prefix or f"stability_{args.year}_{datetime.now():%Y%m%d-%H%M%S}"
    if args.reconstruct_dir:
        processed_log = args.processed_log if os.path.exists(args.processed_log) else None
        results = reconstruct_stability(
            geocoded_dir=args.reconstruct_dir,
            year=args.year,
            key_column=args.key,
            processed_log=processed_log,
            order=args.order,
            extra_columns=args.compare_column,
        )
    else:
        if not args.old or not args.new:
            parser.error("Either pass --reconstruct-dir, or pass both --old and --new.")
        results = compare_snapshots(
            old_path=args.old,
            new_path=args.new,
            year=args.year,
            key_column=args.key,
            extra_columns=args.compare_column,
        )
    written = write_outputs(results, args.outdir, prefix)

    if "summary" in results:
        summary = results["summary"].iloc[0].to_dict()
        print(
            "Stability audit complete: "
            f"{summary['old_cases']} -> {summary['new_cases']} cases in {args.year}; "
            f"{summary['added_cases']} added, {summary['removed_cases']} removed, "
            f"{summary['changed_existing_cases']} existing cases changed."
        )
    else:
        final = results["timeline"].iloc[-1].to_dict()
        print(
            "Reconstructed stability audit complete: "
            f"{int(final['new_cases'])} cases in {args.year} after "
            f"{int(final['step'])} files."
        )
    for name, path in written.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
