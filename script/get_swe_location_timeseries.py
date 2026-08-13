#!/usr/bin/env python3

import sys
from lib.get_location_value import extract_swe_timeseries, plot_timeseries


def main():

    if len(sys.argv) < 5:
        print("Usage:")
        print("python script/extract_swe_timeseries.py <swe_folder> <lon> <lat> <station_name>")
        sys.exit(1)

    swe_folder = sys.argv[1]
    lon = float(sys.argv[2])
    lat = float(sys.argv[3])
    station = sys.argv[4]

    print("Inside extract_swe_timeseries.py")

    try:
        df = extract_swe_timeseries(
            swe_folder,
            lon,
            lat
        )

        print(df)

        plot_timeseries(
            df,
            station
        )

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("SWE time series extraction completed successfully.")


if __name__ == "__main__":
    main()