import sys
from GeocodeClass import Geocoder  # Ensure your class is in this file or imported


def main(input_csv):
    geocoder = Geocoder()
    geocoder.geocode(input_csv)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python geocode.py <input_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    main(input_csv)
