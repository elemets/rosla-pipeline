import sys
from GeocodeClass import Geocoder
import argparse


def main(input_csv):
    geocoder = Geocoder()
    geocoder.geocode(input_csv, output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geocode addresses in a CSV file.")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file.")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file.")

    args = parser.parse_args()
    input_csv = args.input
    output_csv = args.output

    main(input_csv)
