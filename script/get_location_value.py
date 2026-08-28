#!/usr/bin/env python3

import sys
from lib.get_location_value import get_pixel_value


def main():

    if len(sys.argv) < 4:
        print("Usage:")
        print("python scripts/get_location_value.py <file_path.tif> <lon> <lat>")
        sys.exit(1)

    file_path = sys.argv[1]
    lon = float(sys.argv[2])
    lat = float(sys.argv[3])

    print("inside script get_location_value.py")

    try:
        value, row, col = get_pixel_value(file_path, lon, lat)
        print(f"Result -> value={value}, row={row}, col={col}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("get_location_value completed successfully.")


if __name__ == "__main__":
    main()