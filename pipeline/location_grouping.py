import sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import argparse
import os
import numpy as np

columns_to_check = [
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
    "EventAddress",
]


def coalesce_duplicate_rows(df, subset, tie_keep="first"):
    """
    Collapse rows that share the same value(s) in `subset` into a single row.

    Instead of arbitrarily keeping the first/last duplicate, the row with the
    most non-null values is used as the base, and any remaining nulls are then
    filled in using values from the other rows in the same group.

    Parameters:
    - df (pd.DataFrame): The DataFrame to deduplicate.
    - subset (str or list): Column name(s) that identify a duplicate.
    - tie_keep (str): Which row wins for cells where equally complete rows
      hold *different* non-null values. "first" keeps the earlier row's
      value, "last" keeps the later row's value (mirrors drop_duplicates'
      keep argument).

    Returns:
    - pd.DataFrame: One row per unique `subset` value, with nulls minimized.
    """
    original_columns = df.columns.tolist()

    # Rank rows by how complete they are (most non-null values first).
    completeness = df.notna().sum(axis=1)
    df_ranked = df.assign(_completeness=completeness)

    # Among equally complete rows a *stable* sort preserves the existing row
    # order, so the first row would win ties. Reverse first when the caller
    # wants the last row to win instead.
    if tie_keep == "last":
        df_ranked = df_ranked[::-1]

    df_sorted = df_ranked.sort_values(
        "_completeness", ascending=False, kind="stable"
    ).drop(columns="_completeness")

    # groupby().first() takes the first *non-null* value in each column, so
    # the base (most complete) row's values win and any gaps are filled from
    # the remaining rows in the group.
    combined = df_sorted.groupby(
        subset, as_index=False, sort=False, dropna=False
    ).first()

    # Restore the original column order (groupby moves the key columns first).
    return combined[original_columns]


def join_census_tract(gdf_points, census_geojson):
    # Load the census tracts GeoJSON file
    gdf_census = gpd.read_file(census_geojson)

    # Ensure CRS match
    if gdf_census.crs != gdf_points.crs:
        gdf_census = gdf_census.to_crs(gdf_points.crs)

    # Perform spatial join
    gdf_joined = gpd.sjoin(
        gdf_points,
        gdf_census,
        how="left",
        predicate="within",
        lsuffix="_point",
        rsuffix="_census",
    )
    # Drop the index_right column to prevent conflicts in subsequent joins
    if "index_right" in gdf_joined.columns:
        gdf_joined = gdf_joined.drop(columns=["index_right"])

    return gdf_joined


def join_zip_codes(gdf_points, zip_geojson):
    # Drop 'index_right' if it exists to avoid conflicts
    if "index_right" in gdf_points.columns:
        gdf_points = gdf_points.drop(columns=["index_right"])
    if "index_left" in gdf_points.columns:
        gdf_points = gdf_points.drop(columns=["index_left"])

    # Load the zip codes GeoJSON file
    gdf_zip = gpd.read_file(zip_geojson)

    # Ensure CRS match
    if gdf_zip.crs != gdf_points.crs:
        gdf_zip = gdf_zip.to_crs(gdf_points.crs)

    # Perform spatial join
    gdf_joined = gpd.sjoin(gdf_points, gdf_zip, how="left", predicate="within")

    # Drop 'index_right' after the join to clean up
    if "index_right" in gdf_joined.columns:
        gdf_joined = gdf_joined.drop(columns=["index_right"])

    return gdf_joined


def joining_similar_columns(dataframe):

    column_groups = {
        "CaseNumber": ["CaseNum", "CaseNumber", "Case Number"],
        "ResidenceType": ["ResType", "ResidenceType", "Residence Type"],
        "DeathDate": ["DeathDate", "Date of Death", "Death Date"],
        "DeathTime": ["DeathTime", "Time of Death", "Death Time"],
        "DeathAddress": [
            "DeathAddr",
            "DeathAdress",
            "DeathAddr.1",
            "address.death",
            "DeathAdress.1",
        ],
        "Race": ['Races', "race", "Race"],
        "DeathZip": ["DeathZip", "DeathZip.1"],
        "EventPlace": ["EventPlace", "Event Place"],
        "EventAddress": ["EventAddr", "EventAddr.1", "eventaddress", "Event Address"],
        "EventZip": ["EventZip", "EventZip.1"],
        "Mode": ["Mode", "Mode.1"],
        "CauseA": ["CauseA", "Cause A"],
        "CauseB": ["CauseB", "Cause B"],
        "CauseC": ["CauseC", "Cause C"],
        "CauseD": ["CauseD", "Cause D"],
        "CauseOther": ["OtherCause", "Other Cause"],
        "HowInjuryOccurred": ["HowInjuryOccurred", "InjuryDesc"],
        "FirstName": ["First Name"],
        "MiddleName": ["Middle Name"],
        "LastName": ["Last Name"],
        "DateOfBirth": ["Date of Birth", "BirthDate", "DateofBirth"],
        "Text": ["text"],
        "Address": ["address"],  # Need to verify what this refers to
        # Add any other columns as needed
    }

    for new_col, old_cols in column_groups.items():
        # Find which of the old columns exist in the DataFrame
        existing_cols = [col for col in old_cols if col in dataframe.columns]
        if not existing_cols:
            continue  # No columns to merge for this group
        # Combine columns into the new column
        dataframe[new_col] = dataframe[existing_cols].bfill(axis=1).iloc[:, 0]
        # Drop the old columns
        dataframe.drop(
            columns=[col for col in existing_cols if col != new_col], inplace=True
        )
    return dataframe


def parse_dates(row):
    date_formats = ["%m/%d/%Y", "%Y-%m-%d"]
    for fmt in date_formats:
        try:
            return pd.to_datetime(row["DeathDate"], format=fmt)
        except ValueError:
            continue
    # If all formats fail, return NaT
    return pd.NaT


def clean_and_categorize_race(race):
    if pd.isna(race):
        return np.nan

    race = race.lower()           # Case folding
    race = race.replace(" ", "")  # Remove spaces
    race = race.replace("\n", "") # Remove newline characters

    # If multiple races are listed in one row, pick the first
    if "," in race:
        parts = [p for p in race.split(",") if p not in ("unknown/other", "unknown", "null")]
        race = parts[0] if parts else "unknown/other"
        
    if race in ["americanindian", "nativeamerican"]:
        return "AMERICAN INDIAN"
    elif race in ["armenian", "middleeastern"]:
        return "MIDDLE EASTERN"
    elif race in [
        "asian",
        "cambodian",
        "chinese",
        "filipino",
        "japanese",
        "korean",
        "eastindian",
        "thai",
        "vietnamese",
    ]:
        return "ASIAN"
    elif race in ["black"]:
        return "BLACK"
    elif race in [
        "guamanian",
        "hawaiian",
        "pacificislander",
        "samoan",
        "tongan",
        "nativehawaiian/otherpacificislander",
    ]:
        return "PACIFIC ISLANDER"
    elif race in ["hispanic/latino", "hispanic/latina", "hispanic/latinamerican"]:
        return "LATINE"
    elif race in ["white", "caucasian", "white/caucasian"]:
        return "WHITE"
    elif race in ["unknown", "null", "unknown/other"]:
        return np.nan
    else:
        return race
    
def final_clean(current_df):


    valid_races = [
    "WHITE",
    "LATINE",
    "BLACK",
    "ASIAN",
    "MIDDLE EASTERN",
    "AMERICAN INDIAN",
    "PACIFIC ISLANDER",
    "UNKNOWN"
    ]


    gender_map = {
        "M": "MALE",
        "MALE": "MALE",
        "F": "FEMALE",
        "FEMALE": "FEMALE",
        "NON-BINARY": "NON-BINARY"
    }

    current_df["Mode"] = (
        current_df["Mode"]
        .str.replace("\n", "", regex=False)   
        .str.strip()                          
        .str.upper()                          
    )

    current_df["Mode"] = current_df["Mode"].replace({
        "UNDETERMI": "UNDETERMINED"
    })

    # standardize then filter
    current_df["Race"] = (
        current_df["Race"]
        .str.replace("\n", "", regex=False)
        .str.strip()
        .str.upper()
    )

    current_df.loc[~current_df["Race"].isin(valid_races), "Race"] = np.nan

    current_df["Gender"] = (
        current_df["Gender"]
        .str.replace("\n", "", regex=False)
        .str.strip()
        .str.upper()
    )


    current_df["Gender"] = current_df["Gender"].map(gender_map)
    
    return current_df

def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Group data by location.")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file.")
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Output directory for the final CSV."
    )
    parser.add_argument("-n", "--dropname", required=False, default=True, type=str2bool)
    args = parser.parse_args()

    input_file = args.input
    output_dir = args.output_dir
    dropname = args.dropname

    # Read the input CSV file
    df = pd.read_csv(input_file)

    # Check if 'longitude' and 'latitude' columns exist
    if "lon" not in df.columns or "lat" not in df.columns:
        print("Error: Input CSV must contain 'longitude' and 'latitude' columns.")
        sys.exit(1)

    # Create Point geometries from longitude and latitude
    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry)

    # Set the coordinate reference system (CRS) to WGS84 (EPSG:4326)
    gdf_points.set_crs(epsg=4326, inplace=True)

    census_geojson = "../data/censustracts.geojson"
    zip_geojson = "../data/zipcodes.geojson"
    # Join with census tracts
    gdf_with_census = join_census_tract(gdf_points, census_geojson)

    # Join with zip codes
    gdf_with_zip = join_zip_codes(gdf_with_census, zip_geojson)

    # ### before joining similar columns we want to drop duplicates in the different

    gdf_merged = gdf_with_zip
    gdf_merged["DeathDate"] = gdf_merged.apply(parse_dates, axis=1)

    unparsed_dates = gdf_merged[gdf_merged["DeathDate"].isna()]
    if not unparsed_dates.empty:
        print("These dates couldn't be parsed:")
        print(unparsed_dates)

    # Format the dates uniformly as 'YYYY-MM-DD'
    gdf_merged["DeathDate"] = gdf_merged["DeathDate"].dt.strftime("%Y-%m-%d")

    # Determine oldest and newest 'DeathDate' in the data
    gdf_merged["DeathDate_parsed"] = pd.to_datetime(
        gdf_merged["DeathDate"], errors="coerce"
    )
    valid_dates = gdf_merged["DeathDate_parsed"].dropna()

    if not valid_dates.empty:
        oldest_date = valid_dates.min()
        newest_date = valid_dates.max()

        oldest_str = oldest_date.strftime("%Y-%m")
        newest_str = newest_date.strftime("%Y-%m")

        # Construct the output filename
        output_filename = f"{oldest_str}-{newest_str}-overdoses.csv"
        output_path = os.path.join(output_dir, output_filename)
    else:
        print(
            "No valid 'DeathDate' values found. Using default output filename 'final.csv'."
        )
        output_path = os.path.join(output_dir, "final.csv")

    gdf_merged = gdf_merged.reset_index(drop=True)

    #### Merge racial categories
    gdf_merged = joining_similar_columns(gdf_merged)

    gdf_merged["Race"] = gdf_merged["Race"].apply(clean_and_categorize_race)

    #### Dropping irrelevant columns
    if dropname:
        gdf_merged.drop(
            columns=[
                "DeathAddr",
                "EventAddr",
                "address.death",
                "eventaddress",
                "address",
                "EventCityDesc.1",
                "Unnamed: 0",
                "Zip",
                "Zip.1",
                "FirstName",
                "MiddleName",
                "LastName",
                "CauseC",
                "CauseD",
                "Unnamed: 0.1",
                "Unnamed: 0",
                "address.death",
                "index__census",
                "OBJECTID_left",
                "Races",
                "OBJECTID_right",
            ],
            inplace=True,
            errors="ignore",
        )
    else:
        gdf_merged.drop(
            columns=[
                "DeathAddr",
                "EventAddr",
                "address.death",
                "eventaddress",
                "address",
                "EventCityDesc.1",
                "Unnamed: 0",
                "Zip",
                "Zip.1",
                "CauseC",
                "CauseD",
                "Unnamed: 0.1",
                "Unnamed: 0",
                "address.death",
                "index__census",
                "OBJECTID_left",
                "Races",
                "OBJECTID_right",
            ],
            inplace=True,
            errors="ignore",
        )

    ### Age clean up
    gdf_merged["Age"] = gdf_merged["Age"].str.extract("(\d+)").astype(float)

    # Collapse duplicate "CaseNumber" rows: keep the most complete row and
    # fill its remaining nulls from the other duplicate rows. tie_keep="last"
    # preserves the previous keep="last" preference when two equally complete
    # rows hold conflicting values.
    gdf_merged = coalesce_duplicate_rows(
        gdf_merged, subset="CaseNumber", tie_keep="last"
    )

    gdf_merged = final_clean(gdf_merged)

    gdf_merged = gdf_merged.sort_values(by="DeathDate_parsed", ascending=True)

    gdf_merged.drop(columns=["geometry", "DeathDate_parsed"]).to_csv(
        output_path, index=False
    )