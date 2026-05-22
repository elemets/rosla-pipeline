"""
One-off: rebuild the master geocoded file from the individual *_geocoded.csv
files using the current deduplication logic (coalesce_duplicate_rows +
fill_from_historical).

Run this once after changing the dedup logic. It does NOT re-geocode anything --
it only re-merges the per-file geocoded artifacts that already exist on disk.
Afterwards, rerun location_grouping.py on the rebuilt master to refresh the
final grouped output.
"""

import os
import glob
import shutil
from datetime import datetime

import pandas as pd

# Reuse the exact same logic the live pipeline uses, so the rebuilt master is
# identical to what the incremental pipeline would produce.
# NOTE: adjust the module name below if your orchestrator file is not pipeline.py
from pipeline import (
    OUTPUTDIR,
    coalesce_duplicate_rows,
    fill_from_historical,
    load_historical_data,
)


def rebuild_master_geocoded():
    geocode_dir = os.path.join(OUTPUTDIR, "geocoded")
    master_file = os.path.join(geocode_dir, "combined_classified_data_geocoded.csv")

    # Gather every per-file geocoded artifact, excluding the master itself.
    individual_files = [
        f
        for f in glob.glob(os.path.join(geocode_dir, "*_geocoded.csv"))
        if os.path.abspath(f) != os.path.abspath(master_file)
    ]

    if not individual_files:
        print(f"No '*_geocoded.csv' files found in {geocode_dir}. Nothing to rebuild.")
        return

    print(f"Found {len(individual_files)} individual geocoded files.")

    # Back up the existing master so the old version stays recoverable.
    if os.path.exists(master_file):
        backup = f"{master_file}.bak-{datetime.now():%Y%m%d-%H%M%S}"
        shutil.copy2(master_file, backup)
        print(f"Backed up existing master to: {backup}")

    # Concatenate all per-file geocoded data.
    frames = []
    for f in individual_files:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
            print(f"  loaded {f}")
        except Exception as e:
            print(f"  WARNING: could not read {f}: {e}")

    if not frames:
        print("No geocoded files could be read. Aborting without touching the master.")
        return

    combined_df = pd.concat(frames, ignore_index=True)
    print(f"Combined shape before dedup: {combined_df.shape}")

    # Same two-step merge as append_to_master_geocoded:
    #   1. collapse duplicate CaseNumbers, keeping the most complete row and
    #      filling its nulls from the other duplicate rows.
    #   2. fill any remaining nulls from historical records. In a full rebuild
    #      this is effectively a safety net (the "historical" files are the
    #      same set being merged), but it keeps the result identical to what
    #      the incremental pipeline produces.
    if "CaseNumber" in combined_df.columns:
        combined_df = coalesce_duplicate_rows(
            combined_df, subset="CaseNumber", tie_keep="last"
        )
        historical_data = load_historical_data(geocode_dir)
        combined_df = fill_from_historical(combined_df, historical_data)
        print(f"Shape after dedup: {combined_df.shape}")
    else:
        print("WARNING: no 'CaseNumber' column found; writing combined data as-is.")

    combined_df.to_csv(master_file, index=False)
    print(f"\nRebuilt master written to: {master_file}")
    print("Next: rerun location_grouping.py on this file to refresh the final output.")


if __name__ == "__main__":
    rebuild_master_geocoded()