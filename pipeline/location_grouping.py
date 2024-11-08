import sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import argparse
import os

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
    "Drug No Opioids",
    "EventAddress",
]


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
        "DeathZip": ["DeathZip", "DeathZip.1"],
        "EventPlace": ["EventPlace", "Event Place"],
        "EventAddress": ["EventAddr", "EventAddr.1", "eventaddress", "Event Address"],
        "EventZip": ["EventZip", "EventZip.1"],
        "Mode": ["Mode", "Mode.1"],
        "CauseA": ["CauseA", "Cause A"],
        "CauseB": ["CauseB", "Cause B"],
        "CauseC": ["CauseC", "Cause C"],
        "CauseD": ["CauseD", "Cause D"],
        "CauseOther": ["CauseOther", "Other Cause"],
        "HowInjuryOccurred": ["HowInjuryOccurred", "InjuryDesc"],
        "FirstName": ["First Name"],
        "MiddleName": ["Middle Name"],
        "LastName": ["Last Name"],
        "DateOfBirth": ["Date of Birth"],
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


def resolve_duplicates(group):
    if len(group) == 1:
        return group

    for col in columns_to_check[:-1]:  # Exclude 'EventAddress' for now
        max_value = group[col].max()
        group = group[group[col] == max_value]
        if len(group) == 1:
            return group

    group["EventAddress_length"] = group["EventAddress"].astype(str).str.len()
    max_length = group["EventAddress_length"].max()
    group = group[group["EventAddress_length"] == max_length]
    group = group.drop(columns="EventAddress_length")

    return group.iloc[[0]]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Group data by location.")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file.")
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Output directory for the final CSV."
    )
    args = parser.parse_args()

    input_file = args.input
    output_dir = args.output_dir

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

    gdf_merged = joining_similar_columns(gdf_with_zip)
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

    processed_df = gdf_merged.groupby("CaseNumber", group_keys=False).apply(
        resolve_duplicates
    )

    processed_df = processed_df.reset_index(drop=True)

    processed_df.drop(columns=["geometry", "DeathDate_parsed"]).to_csv(
        output_path, index=False
    )
