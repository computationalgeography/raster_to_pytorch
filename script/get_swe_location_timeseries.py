#!/usr/bin/env python3

import sys
from lib.get_location_value import extract_swe_timeseries, plot_timeseries


def main():

    if len(sys.argv) < 5:
        print("Usage:")
        print("python script/extract_swe_timeseries.py <swe_folder> <lon> <lat> <station_name> [starte_date] [end_date]")
        sys.exit(1)

    swe_folder = sys.argv[1]
    lon = float(sys.argv[2])
    lat = float(sys.argv[3])
    station = sys.argv[4]
    
    start_date = None
    end_date = None

    if len(sys.argv) > 5:
        start_date = None if sys.argv[5] == "None" else sys.argv[5]

    if len(sys.argv) > 6:
        end_date = None if sys.argv[6] == "None" else sys.argv[6]
        

    try:
#        df = extract_swe_timeseries(
#            swe_folder,
#            lon,
#            lat
#        )

        
        df = extract_swe_timeseries(
            swe_folder,
            lon,
            lat,
            start_date,
            end_date
        )

        print(df)

        # Save the DataFrame to a CSV file
        output_csv = f"{station}_swe_timeseries.csv"
        df.to_csv(output_csv, index=False)

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