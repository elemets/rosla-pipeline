import pandas as pd

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

# --- 2. Aggregate overdose counts by ZIPCODE AND Year ---
df_subset = latest_df[["ZIPCODE", "Year"] + drug_cols].copy()
df_agg = df_subset.groupby(["ZIPCODE", "Year"])[drug_cols].sum().reset_index()

# Rename columns for clarity (e.g., Alcohol -> Alcohol_Count)
df_agg.columns = ["ZIPCODE", "Year"] + [f"{col}_Count" for col in drug_cols]

# --- 3. Create the Long Table ---
# Melt the aggregated DataFrame so that each row represents one drug type's count for a given ZIPCODE and Year.
long_table = df_agg.melt(
    id_vars=["ZIPCODE", "Year"],
    value_vars=[f"{col}_Count" for col in drug_cols],
    var_name="Overdose_Type",
    value_name="Overdose_Count",
)

# --- 4. Add a Composite Key ---
# This key combines ZIPCODE, Year, and Overdose_Type.
long_table["composite_key"] = (
    long_table["ZIPCODE"].astype(str)
    + "_"
    + long_table["Year"].astype(str)
    + "_"
    + long_table["Overdose_Type"]
)

# --- 5. Save the Table for Filtering Charts ---
long_table.to_csv("long_table_for_charts.csv", index=False)
