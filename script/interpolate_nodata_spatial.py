import sys
import os
import re
import csv
from datetime import datetime

from lib.interpolate_nodata_fill import fill_nodata_spatial


def extract_info(filepath):
    """
    Extract variable and date from a CHELSA TIFF filename.

    Example:
        CHELSA_pr_15_07_2006_V.2.1.tif
        -> variable = 'pr'
        -> date = 2006-07-15
    """

    filename = os.path.basename(filepath)

    match = re.search(
        r"CHELSA_([^_]+)_(\d{2})_(\d{2})_(\d{4})_V",
        filename
    )

    if match is None:
        raise ValueError(
            f"Could not extract date/variable from: {filepath}"
        )

    variable = match.group(1)

    day = int(match.group(2))
    month = int(match.group(3))
    year = int(match.group(4))

    date = datetime(year, month, day).date()

    return variable, date


def read_file_paths(csv_file):
    """
    Read TIFF paths from the 'path' column of the CSV file.
    """

    file_paths = []

    with open(csv_file, "r", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            filepath = row["path"].strip()

            if filepath:
                file_paths.append(filepath)

    return file_paths


def main():

    if len(sys.argv) != 5:

        print(
            "Usage: python run_nodata_fill.py "
            "<csv_file> <output_path> <max_distance> <smoothing_iterations>"
        )

        sys.exit(1)

    csv_file = sys.argv[1]
    output_path = sys.argv[2]
    max_distance = int(sys.argv[3])
    smoothing_iterations = int(sys.argv[4])

    # ---------------------------------------------------------
    # Read TIFF paths from CSV
    # ---------------------------------------------------------

    file_paths = read_file_paths(csv_file)

    print(
        f"Total files in CSV: {len(file_paths)}"
    )

    # ---------------------------------------------------------
    # Process every file
    # ---------------------------------------------------------

    for current_file in file_paths:

        variable, date = extract_info(
            current_file
        )

        # Output directory:
        #
        # output_path/
        #     pr/
        #     rsds/
        #     ...
        #

        variable_output_path = os.path.join(
            output_path,
            variable
        )

        os.makedirs(
            variable_output_path,
            exist_ok=True
        )

        filename = os.path.basename(
            current_file
        )

        output_file = os.path.join(
            variable_output_path,
            filename
        )

        print()
        print("Processing:")
        print(f"  Variable : {variable}")
        print(f"  Date     : {date}")
        print(f"  Input    : {current_file}")
        print(f"  Output   : {output_file}")

        # -----------------------------------------------------
        # Spatial NoData interpolation
        # -----------------------------------------------------

        fill_nodata_spatial(
            current_file,
            output_file,
            max_distance=max_distance,
            smoothing_iterations=smoothing_iterations,
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()