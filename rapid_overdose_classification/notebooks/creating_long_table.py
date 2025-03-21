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

# Columns to keep in the detailed table (adjust as needed)
# e.g. ZIPCODE, Year, Age, Gender, Race, etc.
id_vars = ["ZIPCODE", "Year", "Age_Bin", "Age", "Gender", "Race"]

# 1. Melt raw data so each row is (Case x Drug)
long_df = latest_df.melt(
    id_vars=id_vars,
    value_vars=drug_cols,
    var_name="Overdose_Type",
    value_name="Occurred",
)

# 2. Keep only rows where the drug was involved
long_df = long_df[long_df["Occurred"] == 1].drop(columns=["Occurred"])

# 3. Create the composite key for normal (per-year) records
long_df["composite_key"] = (
    long_df["ZIPCODE"].astype(str)
    + "_"
    + long_df["Year"].astype(str)
    + "_"
    + long_df["Overdose_Type"]
    + "_Count"
)

# 4. Duplicate each record as "All" year
long_df_all = long_df.copy()
long_df_all["Year"] = "All"
long_df_all["composite_key"] = (
    long_df_all["ZIPCODE"].astype(str)
    + "_All_"
    + long_df_all["Overdose_Type"]
    + "_Count"
)

# 5. Combine the per-year records + the all-year records
long_df_final = pd.concat([long_df, long_df_all], ignore_index=True)

# 6. Save the detailed table (CSV) for your charts
long_df_final.to_csv("detailed_long_table_with_all_years.csv", index=False)
