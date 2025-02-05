import time
import pandas as pd
from typing import Optional, Tuple
import geopandas as gp
import re
from geopy.geocoders import Nominatim, ArcGIS
from geopy.exc import GeocoderTimedOut, GeocoderQuotaExceeded
from unidecode import unidecode
from shapely.geometry import Point
from tqdm import tqdm

"""
Geocoding class, this class uses Nominatim (openstreetmaps) and ArcGIS to translate
from a string which contains an address into a longitude and latitude. Extensive
cleaning is done first and the lon and lat are extracted into separate columns 
ready for input into the next step of the pipeline. 
"""


class Geocoder:
    def __init__(self) -> None:
        tqdm.pandas()
        pass

    def geocode(self, input_csv: str, output_csv: str) -> None:
        non_geocoded_df = pd.read_csv(input_csv)
        # Replace 'NULL' strings with NaN
        non_geocoded_df = non_geocoded_df.replace("NULL", pd.NA)

        non_geocoded_df_cols = non_geocoded_df.columns

        possible_death_add_names = [
            "DeathAddress",
            "DeathAdress",
            "Death Address",
            "Event Address",
            "EventAddress",
        ]

        ### Trying to normalise the data so we can access the address columns
        if any(x in non_geocoded_df_cols for x in possible_death_add_names):
            non_geocoded_df = non_geocoded_df.rename(
                columns={"Death Address": "DeathAddress", "Death Zip Code": "DeathZip"}
            )
            non_geocoded_df = non_geocoded_df.rename(
                columns={"DeathAddress": "DeathAddress", "Death Zip Code": "DeathZip"}
            )
            non_geocoded_df = non_geocoded_df.rename(
                columns={"DeathAddr": "DeathAddress"}
            )
            non_geocoded_df = non_geocoded_df.rename(
                columns={
                    "Event Address": "EventAddress",
                    "Event Zip": "EventZip",
                    "EventCityDesc": "EventCity",
                }
            )
            non_geocoded_df = non_geocoded_df.rename(
                columns={
                    "EventAddr": "EventAddress",
                    "Event Zip": "EventZip",
                    "EventCityDesc": "EventCity",
                }
            )

        # Combine columns into single columns and handle NaN values
        non_geocoded_df["address.death"] = non_geocoded_df[
            ["DeathAddress", "DeathCity", "DeathZip"]
        ].apply(lambda x: ", ".join(x.dropna().astype(str)), axis=1)
        non_geocoded_df["event.address"] = non_geocoded_df[
            ["EventAddress", "EventCity", "EventZip"]
        ].apply(lambda x: ", ".join(x.dropna().astype(str)), axis=1)

        # Replace 'UNKNOWN' and empty strings with NaN, clean addresses, and apply replacements
        non_geocoded_df["event.address"] = non_geocoded_df["event.address"].apply(
            lambda x: pd.NA if "unknown" in x.lower() else x
        )
        non_geocoded_df.replace({"UNKNOWN": pd.NA, "": pd.NA}, inplace=True)
        non_geocoded_df.replace({"UNKNOWN, UNKNOWN": pd.NA, "": pd.NA}, inplace=True)
        non_geocoded_df.replace({"UNK": pd.NA, "": pd.NA}, inplace=True)

        # using address cleaner function which is defined above
        non_geocoded_df[["event.address", "address.death"]] = non_geocoded_df[
            ["event.address", "address.death"]
        ].applymap(self._address_cleaner)

        # Using regex to replace patterns
        non_geocoded_df[["event.address", "address.death"]] = non_geocoded_df[
            ["event.address", "address.death"]
        ].replace(to_replace=[" #.*,", "UNIT", "APT"], value=["", "", ""], regex=True)

        # Fill NaN in event.address with address.death values
        non_geocoded_df["event.address"] = non_geocoded_df["event.address"].fillna(
            non_geocoded_df["address.death"]
        )

        non_geocoded_df["event.address"] = non_geocoded_df["event.address"].str.replace(
            r"\.0$", "", regex=True
        )

        non_geocoded_df = non_geocoded_df.rename(
            columns={"event.address": "eventaddress"}
        )

        non_geocoded_df["geometry"] = pd.NA

        print("Geocoding the data")
        for index, row in tqdm(
            non_geocoded_df.iterrows(), total=non_geocoded_df.shape[0]
        ):
            lat, lon, address = self.geocode_address(
                row["eventaddress"], provider="arcgis"
            )
            if address is not None:  # Update DataFrame if geocoding was successful
                non_geocoded_df.at[index, "geometry"] = Point(lon, lat)
                non_geocoded_df.at[index, "address"] = address

        for geom in non_geocoded_df["geometry"]:
            if type(geom) == float:
                geom = None
                continue

        filtered_df = non_geocoded_df[non_geocoded_df["geometry"].isna()]
        if not filtered_df.empty:
            locations_of_nulls = gp.tools.geocode(
                filtered_df["address.death"], provider="arcgis", timeout=None
            )
            non_geocoded_df.loc[non_geocoded_df["geometry"].isna(), "geometry"] = (
                locations_of_nulls["geometry"]
            )
            non_geocoded_df.loc[non_geocoded_df["geometry"].isna(), "address"] = (
                locations_of_nulls["address"]
            )

        print("Extracting the longitude and latitude:")
        non_geocoded_df["lon"] = non_geocoded_df["geometry"].progress_apply(
            self._extract_lon
        )
        non_geocoded_df["lat"] = non_geocoded_df["geometry"].progress_apply(
            self._extract_lat
        )

        non_geocoded_df.to_csv(output_csv, index=False)

    ## function which deals with geocoding and has an added sleep function meaning it
    ## shouldnt time out
    def geocode_address(
        self, address: str, provider: str = "nominatim"
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:

        time.sleep(0.2)
        if provider == "nominatim":
            geolocator = Nominatim(user_agent="geocodingapp")
        else:  # Default to ArcGIS if not Nominatim
            geolocator = ArcGIS()
        try:
            location = geolocator.geocode(address, timeout=10)
            if location:
                return location.latitude, location.longitude, location.address
            else:
                return None, None, None
        except (GeocoderTimedOut, GeocoderQuotaExceeded) as e:
            print(f"Error geocoding {address}: {e}")
            return None, None, None

    @staticmethod
    def _address_cleaner(address: Optional[str]) -> str:
        if pd.isna(address):
            return address

        address = address.replace("\xa0", " ")
        address = re.sub(r"[\x01-\x1F\x7F]", " ", address)
        address = re.sub(r" +", " ", address).strip()
        address = unidecode(address)
        special_chars = {"½": "1/2", "ª": "a", "º": "o"}
        for char, replacement in special_chars.items():
            address = address.replace(char, replacement)
        address = re.sub(r"[^\x00-\x7F]+", " ", address)
        address = re.sub(r'["\'*]', "", address)
        address = re.sub(r"^,*|(?<=,),|,*$", "", address)
        address = re.sub(r",+", ",", address)
        address = re.sub(r"c/o|c/0|c/", "", address, flags=re.IGNORECASE)

        return address

    @staticmethod
    def _extract_lon(geometry: Point) -> float:
        try:
            x_value = geometry.x
        except:
            x_value = None
        return x_value

    @staticmethod
    def _extract_lat(geometry: Point) -> float:
        try:
            y_value = geometry.y
        except:
            y_value = None
        return y_value
