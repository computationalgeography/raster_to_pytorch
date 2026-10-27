

import sys
import os
import re
import csv
from datetime import datetime

from lib.interpolate_nodata_fill import fill_nodata_spatial

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt


def find_max_distance(input_file, mask_file):
    """
    Find the maximum distance, in pixels, from a masked NoData pixel
    to the nearest valid pixel.

    Only pixels where mask != 0 and input == NoData are considered.
    """

    with rasterio.open(input_file) as src:
        data = src.read(1)
        nodata = src.nodata

    with rasterio.open(mask_file) as src:
        mask = src.read(1)

    if nodata is None:
        raise ValueError(
            f"NoData value is not defined in: {input_file}"
        )

    # Pixels that need to be filled
    missing_masked = (
        (mask != 0) &
        (data == nodata)
    )

    # Pixels that can be used as interpolation sources
    valid = data != nodata

    if not np.any(missing_masked):
        return 0

    if not np.any(valid):
        raise ValueError(
            f"No valid pixels available in: {input_file}"
        )

    # Distance from every pixel to nearest valid pixel
    distance = distance_transform_edt(~valid)

    max_distance = int(
        np.ceil(
            distance[missing_masked].max()
        )
    )

    return max_distance


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
            "<csv_file> <output_path> <mask_file> <smoothing_iterations>"
        )

        sys.exit(1)

    csv_file = sys.argv[1]
    output_path = sys.argv[2]
    mask_file = sys.argv[3]
    smoothing_iterations = int(sys.argv[4])

    file_paths = read_file_paths(csv_file)

    print(f"Total files in CSV: {len(file_paths)}")

    for current_file in file_paths:

        variable, date = extract_info(current_file)

        variable_output_path = os.path.join(
            output_path,
            variable
        )

        os.makedirs(
            variable_output_path,
            exist_ok=True
        )

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

        # -----------------------------------------------------
        # Find required maximum interpolation distance
        # -----------------------------------------------------

        max_distance = find_max_distance(
            current_file,
            mask_file
        )

        print(
            f"  Max distance : {max_distance} pixels"
        )

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