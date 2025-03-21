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


# 2. Prepare data for aggregation
df_subset = latest_df[["ZIPCODE", "Year"] + drug_cols].copy()

# 3. Aggregate by ZIPCODE & Year
df_agg = df_subset.groupby(["ZIPCODE", "Year"])[drug_cols].sum().reset_index()

# 4. Create an "All" row by aggregating across all years for each ZIPCODE
df_all_years = df_subset.groupby("ZIPCODE")[drug_cols].sum().reset_index()
df_all_years["Year"] = "All"

# 5. Combine per-year data with the all-year rows
df_combined = pd.concat([df_agg, df_all_years], ignore_index=True)

# Rename columns (e.g. "Alcohol" -> "Alcohol_Count")
df_combined.columns = ["ZIPCODE", "Year"] + [f"{col}_Count" for col in drug_cols]

# 6. Load ZIP code geometry & simplify
zip_gdf = gpd.read_file("../../data/zipcodes.geojson")
# Simplify geometry with TopoJSON
topo = tp.Topology(zip_gdf, prequantize=False)
simple = topo.toposimplify(0.001).to_gdf()

# Ensure ZIPCODE is a string in both
simple["ZIPCODE"] = simple["ZIPCODE"].astype(str)
df_combined["ZIPCODE"] = df_combined["ZIPCODE"].astype(str)

# 7. Merge geometry with aggregated overdose counts.
# Use the geometry as the base to ensure every zipcode appears.
zip_overdose_gdf = simple.merge(df_combined, on="ZIPCODE", how="left")

# 8. Melt to create one row per Overdose_Type
new_drug_cols = [f"{col}_Count" for col in drug_cols]
df_long = zip_overdose_gdf.melt(
    id_vars=["ZIPCODE", "Year", "OBJECTID", "Shape_Length", "Shape_Area", "geometry"],
    value_vars=new_drug_cols,
    var_name="Overdose_Type",
    value_name="Overdose_Count",
)

# 9. Ensure Year is a string
df_long["Year"] = df_long["Year"].astype(str)

# 10. Build a complete table: ensure every combination of ZIPCODE, Year, and Overdose_Type is present.
# Get all unique zipcodes (from geometry), all years (from df_combined), and all overdose types.
zipcodes = simple["ZIPCODE"].unique()
years = df_combined["Year"].unique()
overdose_types = new_drug_cols  # These already have the "_Count" suffix

# Create a complete combination DataFrame using a MultiIndex
complete_combos = pd.MultiIndex.from_product(
    [zipcodes, years, overdose_types], names=["ZIPCODE", "Year", "Overdose_Type"]
).to_frame(index=False)

# 11. Merge the complete combinations with your melted data.
# Use suffixes to avoid duplicate 'geometry' columns.
df_long_complete = complete_combos.merge(
    df_long, on=["ZIPCODE", "Year", "Overdose_Type"], how="left", suffixes=("", "_drop")
)

# Fill missing overdose counts with 0
df_long_complete["Overdose_Count"] = df_long_complete["Overdose_Count"].fillna(0)

# 12. Merge in the geometry from the simplified ZIP code layer.
# We only need the ZIPCODE and geometry columns from the 'simple' GeoDataFrame.
df_long_complete = df_long_complete.merge(
    simple[["ZIPCODE", "geometry"]], on="ZIPCODE", how="left", suffixes=("", "_geom")
)

# If a geometry column already exists from the previous merge, drop it and rename the geometry from the simple merge.
if (
    "geometry" in df_long_complete.columns
    and "geometry_geom" in df_long_complete.columns
):
    df_long_complete = df_long_complete.drop(columns=["geometry"])
    df_long_complete = df_long_complete.rename(columns={"geometry_geom": "geometry"})
elif "geometry_geom" in df_long_complete.columns:
    df_long_complete = df_long_complete.rename(columns={"geometry_geom": "geometry"})

# 13. Add composite key: ZIPCODE_Year_Overdose_Type (e.g., "90001_2012_Alcohol_Count")
df_long_complete["composite_key"] = (
    df_long_complete["ZIPCODE"].astype(str)
    + "_"
    + df_long_complete["Year"].astype(str)
    + "_"
    + df_long_complete["Overdose_Type"]
)

# 14. Convert to a GeoDataFrame (if not already) and save as GeoJSON.
df_long_complete_gdf = gpd.GeoDataFrame(
    df_long_complete, geometry="geometry", crs=simple.crs
)

df_long_complete_gdf.to_file(
    "zip_overdose_with_all_years_complete_really_no_nans.geojson", driver="GeoJSON"
)
