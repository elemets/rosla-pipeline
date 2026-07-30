"""
Geocode the classified files, reusing coordinates already obtained in a previous
run instead of re-querying ArcGIS for addresses we have seen before.

The cache is keyed on the cleaned address string that GeocodeClass itself passes
to `geocode_address`, so a hit returns exactly what the live call would have.
Anything not in the cache (new cases, addresses changed by the re-extraction) is
geocoded for real, with the same 0.2s pacing as before.

Usage: python geocode_cached.py [--dry-run]
"""

import glob
import os
import sys

import pandas as pd

sys.path.insert(0, "/home/afunnell/Code/Rapid_overdose_clean/pipeline")
os.chdir("/home/afunnell/Code/Rapid_overdose_clean/pipeline")

from GeocodeClass import Geocoder  # noqa: E402
from pipeline import join_similar_columns_for_file  # noqa: E402

BACKUP = "backups_20260729_rerun/geocoded"
CLASSIFIED = "pipeline_steps/input_files/classified"
GEOCODED = "pipeline_steps/input_files/geocoded"


def build_cache():
    """address string -> (lat, lon, formatted_address) from the previous run."""
    cache = {}
    for path in sorted(glob.glob(os.path.join(BACKUP, "*_geocoded.csv"))):
        if "combined_classified" in path:
            continue
        cols = ["eventaddress", "address.death", "address", "lat", "lon",
                "used_death_address"]
        df = pd.read_csv(path, usecols=lambda c: c in cols, dtype=str,
                         low_memory=False)
        if "lat" not in df.columns:
            continue
        df = df[df["lat"].notna()]
        used_death = df.get("used_death_address", pd.Series("", index=df.index))
        used_death = used_death.astype(str).str.lower().eq("true")
        # The stored result belongs to whichever address actually resolved.
        key_col = df["eventaddress"].where(~used_death, df.get("address.death"))
        for key, lat, lon, addr in zip(key_col, df["lat"], df["lon"], df["address"]):
            if pd.isna(key) or key in cache:
                continue
            try:
                cache[key] = (float(lat), float(lon), addr)
            except (TypeError, ValueError):
                continue
    return cache


class CachedGeocoder(Geocoder):
    def __init__(self, cache):
        super().__init__()
        self.cache = cache
        self.hits = 0
        self.misses = 0

    def geocode_address(self, address, provider="nominatim"):
        if address in self.cache:
            self.hits += 1
            return self.cache[address]
        self.misses += 1
        return super().geocode_address(address, provider=provider)


def main():
    dry = "--dry-run" in sys.argv
    cache = build_cache()
    print(f"cache: {len(cache):,} distinct addresses from the previous run",
          flush=True)

    geo = CachedGeocoder(cache)
    for path in sorted(glob.glob(os.path.join(CLASSIFIED, "*_classified.csv"))):
        base = os.path.basename(path).replace("_classified.csv", "")
        out = os.path.join(GEOCODED, f"{base}_geocoded.csv")
        if os.path.exists(out):
            print(f"SKIP {base} (already geocoded)", flush=True)
            continue
        before_h, before_m = geo.hits, geo.misses
        if dry:
            df = pd.read_csv(path, dtype=str, low_memory=False)
            print(f"DRY  {base}: {len(df):,} rows", flush=True)
            continue
        print(f"START {base}", flush=True)
        # pipeline.py standardises column names between classifying and
        # geocoding; the geocoder needs DeathAddress/EventAddress and their zips
        # to exist under those exact names. Idempotent, so re-running is safe.
        join_similar_columns_for_file(path)
        geo.geocode(path, out)
        print(f"DONE {base}  cache_hits={geo.hits - before_h:,} "
              f"live_calls={geo.misses - before_m:,}", flush=True)

    print(f"TOTAL cache_hits={geo.hits:,} live_calls={geo.misses:,}", flush=True)
    print("ALL_GEOCODING_DONE", flush=True)


if __name__ == "__main__":
    main()
