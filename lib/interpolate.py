import os
import re
from datetime import datetime, timedelta
import numpy as np
import rasterio


def interpolate_missing_day(missing_file_path, output_dir):
    """

    Linearly interpolates a missing daily raster using
    the previous and next day.

    Note:
        Does not support consecutive missing days.

    Parameters
    ----------
    missing_file_path : str
        Full path to missing file
        Example:
        /archive/.../pr/CHELSA_pr_21_03_2022_V.2.1.tif

    output_dir : str
        Directory where filled file will be saved
    """

    input_dir = os.path.dirname(missing_file_path)
    missing_filename = os.path.basename(missing_file_path)

    os.makedirs(output_dir, exist_ok=True)

    # Extract date (e.g. 21_03_2022)
    match = re.search(r"\d{2}_\d{2}_\d{4}", missing_filename)
    if not match:
        raise ValueError("No date found in filename.")

    date_str = match.group()
    date = datetime.strptime(date_str, "%d_%m_%Y")

    prev_date = date - timedelta(days=1)
    next_date = date + timedelta(days=1)

    prev_file = missing_filename.replace(
        date_str, prev_date.strftime("%d_%m_%Y")
    )
    next_file = missing_filename.replace(
        date_str, next_date.strftime("%d_%m_%Y")
    )

    prev_path = os.path.join(input_dir, prev_file)
    next_path = os.path.join(input_dir, next_file)

    if not os.path.exists(prev_path):
       raise FileNotFoundError(
        f"Previous file not found: {prev_path}\n"
        "Possible reason: consecutive missing dates."
    )

    if not os.path.exists(next_path):
       raise FileNotFoundError(
        f"Next file not found: {next_path}\n"
        "Possible reason: consecutive missing dates."
    )

    # Read rasters
    with rasterio.open(prev_path) as src1:
        data1 = src1.read(1)
        profile = src1.profile
        nodata = src1.nodata

    with rasterio.open(next_path) as src2:
        data2 = src2.read(1)

    original_dtype = profile["dtype"]

    # Convert to float to avoid overflow
    data1 = data1.astype("float32")
    data2 = data2.astype("float32")

    avg = (data1 + data2) / 2.0

    # Handle nodata
    if nodata is not None:
        mask = (data1 == nodata) | (data2 == nodata)
        avg[mask] = nodata

    # Convert back safely
    if "int" in original_dtype:
        avg = np.rint(avg)
        info = np.iinfo(np.dtype(original_dtype))
        avg = np.clip(avg, info.min, info.max)
        avg = avg.astype(original_dtype)
    else:
        avg = avg.astype(original_dtype)

    # Save output
    output_path = os.path.join(output_dir, missing_filename)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(avg, 1)

    print(f"Created: {output_path}")
    print(f"Original dtype: {original_dtype}")