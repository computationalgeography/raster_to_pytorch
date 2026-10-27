
import csv
import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

csv_file = "/scratch/depfg/faraj001/20260218_Oriol_snow_model_LSTM/missing_inputs.csv"
mask_file = "/archive/depfg/compgeo/projecto/data/inference_mask.tif"

# ---------------------------------------------------------
# Read mask
# ---------------------------------------------------------

with rasterio.open(mask_file) as src:
    mask = src.read(1)

# ---------------------------------------------------------
# Read file paths from CSV
# ---------------------------------------------------------

with open(csv_file, "r", newline="") as f:
    reader = csv.DictReader(f)
    file_paths = [
        row["path"].strip()
        for row in reader
        if row["path"].strip()
    ]

global_max_distance = 0

# ---------------------------------------------------------
# Process all files
# ---------------------------------------------------------

for i, file_path in enumerate(file_paths, 1):

    print(f"[{i}/{len(file_paths)}] {file_path}")

    with rasterio.open(file_path) as src:
        data = src.read(1)
        nodata = src.nodata

    if nodata is None:
        continue

    # NoData pixels inside the mask
    missing = (
        (mask != 0) &
        (data == nodata)
    )

    if not np.any(missing):
        print("  No masked NoData pixels.")
        continue

    # Valid pixels that can be used for interpolation
    valid = data != nodata

    if not np.any(valid):
        print("  WARNING: no valid pixels.")
        continue

    # Distance to nearest valid pixel
    distance = distance_transform_edt(~valid)

    file_max_distance = int(
        np.ceil(distance[missing].max())
    )

    print(
        f"  Maximum distance: {file_max_distance} pixels"
    )

    global_max_distance = max(
        global_max_distance,
        file_max_distance
    )

# ---------------------------------------------------------
# Final result
# ---------------------------------------------------------

print()
print("==============================")
print(
    "GLOBAL MAXIMUM DISTANCE:",
    global_max_distance,
    "pixels"
)
print("==============================")