import pandas as pd
import geopandas as gpd
import topojson as tp

# 1. Load your CSV which has ZIPCODE, Year, and drug columns
latest_df = pd.read_csv(
    "../../reports/deidentified_overdose_201201202408_zips_0311.csv"
)

drug_cols = [
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
]

# Include 'Year' in the subset if you want per-year counts
df_subset = latest_df[["ZIPCODE", "Year"] + drug_cols].copy()

# 2. Aggregate overdose counts by ZIPCODE AND Year
df_agg = df_subset.groupby(["ZIPCODE", "Year"])[drug_cols].sum().reset_index()

# Rename columns for clarity
df_agg.columns = ["ZIPCODE", "Year"] + [f"{col}_Count" for col in drug_cols]

# 3. Load your ZIP code geometry
zip_gdf = gpd.read_file("../../data/zipcodes.geojson")

# 7. Create a Topology and simplify
topo = tp.Topology(zip_gdf, prequantize=False)
simple = topo.toposimplify(0.01).to_gdf()

simple["ZIPCODE"] = simple["ZIPCODE"].astype(str)
df_agg["ZIPCODE"] = df_agg["ZIPCODE"].astype(str)

# 4. Merge aggregated overdose counts with ZIPCODE geometry
#    This will create multiple rows per ZIPCODE if multiple Years exist.
zip_overdose_gdf = df_agg.merge(simple, on="ZIPCODE", how="left")

# 5. Melt to create a single Overdose_Type column (long format)
new_drug_cols = [f"{col}_Count" for col in drug_cols]

df_long = zip_overdose_gdf.melt(
    id_vars=[
        "ZIPCODE",
        "Year",
        "OBJECTID",  # if it exists in your geojson
        "Shape_Length",  # if it exists
        "Shape_Area",  # if it exists
        "geometry",
    ],
    value_vars=new_drug_cols,
    var_name="Overdose_Type",
    value_name="Overdose_Count",
)

df_long["Year"] = df_long["Year"].astype(str)

df_long_gdf = gpd.GeoDataFrame(
    df_long,
    geometry="geometry",
)

df_long_gdf["composite_key"] = (
    df_long_gdf["ZIPCODE"].astype(str)
    + "_"
    + df_long_gdf["Year"].astype(str)
    + "_"
    + df_long_gdf["Overdose_Type"]
)


# Save final GeoDataFrame
df_long_gdf.to_file("zip_overdose_all_year_drug_long_1_simple.gpkg", driver="GPKG")
