import sys
import os
import re
import csv
from datetime import datetime, timedelta

from lib.interpolate_nodata_fill import nodata_fill_gdal    #nodata_fill_v2


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
        raise ValueError(f"Could not extract date/variable from: {filepath}")

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


def find_non_consecutive_files(file_paths):
    """
    Return files whose date is not consecutive to another
    file of the same variable.

    A file is considered consecutive if another file exists
    exactly one day before or one day after it.
    """

    grouped = {}

    for filepath in file_paths:
        variable, date = extract_info(filepath)

        grouped.setdefault(variable, []).append(
            (date, filepath)
        )

    selected_files = []

    for variable, files in grouped.items():

        dates = {date for date, _ in files}

        for date, filepath in files:

            previous_day = date - timedelta(days=1)
            next_day = date + timedelta(days=1)

            is_consecutive = (
                previous_day in dates
                or next_day in dates
            )

            if not is_consecutive:
                selected_files.append(
                    (date, variable, filepath)
                )

    selected_files.sort(key=lambda x: x[0])

    return selected_files


def main():

    if len(sys.argv) != 5:
        print(
            "Usage: python run_nodata_fill.py "
            "<csv_file> <mask_file> <output_path> <partition_size>"
        )
        sys.exit(1)

    csv_file = sys.argv[1]
    mask_file = sys.argv[2]
    output_path = sys.argv[3]
    partition_size = int(sys.argv[4])

    # Read all paths
    file_paths = read_file_paths(csv_file)

    print(f"Total files in CSV: {len(file_paths)}")

    # Select only non-consecutive files
    selected_files = find_non_consecutive_files(file_paths)

    print(
        f"Files selected for processing: "
        f"{len(selected_files)}"
    )

    for date, variable, current_file in selected_files:

        # Output directory:
        # output_path/pr
        # output_path/rsds
        variable_output_path = os.path.join(
            output_path,
            variable
        )

        os.makedirs(variable_output_path, exist_ok=True)

        filename = os.path.basename(current_file)

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

#        nodata_fill_v2(
#            current_file,
#            mask_file,
#            output_file,
#            partition_size
#        )
        
        nodata_fill_gdal(current_file,
            mask_file,
            output_file,)

    print()
    print("Done.")


if __name__ == "__main__":
    main()