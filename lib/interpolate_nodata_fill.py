#import os
#import lue.framework as lfr
#from osgeo import gdal
#from datetime import datetime, timedelta
#
#
#def nodata_fill(current_file: str, mask_file: str, output_path: str, partition_shape_size: int):
#
#    print("inside function nodata_fill")
#
#    partition_shape: tuple[int, int] = 2 * (partition_shape_size,)
#
#    folder = os.path.dirname(current_file) or "."
#    base = os.path.basename(current_file)
#
#    # --- Parse date ---
#    parts = base.split('_')
#    day = int(parts[2])
#    month = int(parts[3])
#    year = int(parts[4])
#    current_date = datetime(year, month, day)
#
#    prefix = "_".join(parts[:2])   # CHELSA_rsds
#    suffix = "_".join(parts[5:])   # V.2.1.tif
#
#    # --- Build prev / next filenames ---
#    prev_date = current_date - timedelta(days=1)
#    next_date = current_date + timedelta(days=1)
#
#    prev_file = os.path.join(
#        folder,
#        f"{prefix}_{prev_date.day:02d}_{prev_date.month:02d}_{prev_date.year}_{suffix}"
#    )
#
#    next_file = os.path.join(
#        folder,
#        f"{prefix}_{next_date.day:02d}_{next_date.month:02d}_{next_date.year}_{suffix}"
#    )
#
#    # --- Read arrays ---
#    mask = lfr.from_gdal(mask_file, partition_shape=partition_shape)
#    current = lfr.from_gdal(current_file, partition_shape=partition_shape)
#    prev = lfr.from_gdal(prev_file, partition_shape=partition_shape)
#    next = lfr.from_gdal(next_file, partition_shape=partition_shape)
#
#    # --- Build condition ---
#    fill_condition = lfr.logical_and(
#        lfr.logical_not(lfr.valid(current)),
#        mask
#    )
#
#    invalid_prev_next = lfr.logical_or(
#        lfr.logical_not(lfr.valid(prev)),
#        lfr.logical_not(lfr.valid(next))
#    )
#
#    problem_cells = lfr.logical_and(fill_condition, invalid_prev_next)
#
#    # --- Check if any problematic pixel exists ---
#    has_problem = lfr.any(problem_cells)
#
#    if has_problem:
#        raise ValueError(
#            f"Cannot interpolate: some missing pixels in {current_file} "
#            f"have invalid prev/next day data."
#        )
#
#    # --- Compute interpolation ---
#    fill_values = (prev + next) / 2
#
#    result = lfr.where(fill_condition, fill_values, current)
#
#    # --- Save ---
#    output_full_path = os.path.join(output_path, base)
#
#    lfr.to_gdal(result, output_full_path)
#
#    print(f"Interpolation completed. Output saved to {output_full_path}")
#    
#    
#
#def nodata_fill_v2(
#    current_file: str,
#    mask_file: str,
#    output_path: str,
#    partition_shape_size: int,
#):
#
#    print("inside function nodata_fill_v2")
#
#    partition_shape: tuple[int, int] = 2 * (partition_shape_size,)
#
#    folder = os.path.dirname(current_file) or "."
#    base = os.path.basename(current_file)
#
#    # ---------------------------------------------------------
#    # Parse date from filename
#    # Example:
#    # CHELSA_rsds_01_01_2020_V.2.1.tif
#    # ---------------------------------------------------------
#    parts = base.split("_")
#
#    day = int(parts[2])
#    month = int(parts[3])
#    year = int(parts[4])
#
#    current_date = datetime(year, month, day)
#
#    prefix = "_".join(parts[:2])
#    suffix = "_".join(parts[5:])
#
#    # ---------------------------------------------------------
#    # Previous and next file names
#    # ---------------------------------------------------------
#    prev_date = current_date - timedelta(days=1)
#    next_date = current_date + timedelta(days=1)
#
#    prev_file = os.path.join(
#        folder,
#        f"{prefix}_{prev_date.day:02d}_{prev_date.month:02d}_{prev_date.year}_{suffix}",
#    )
#
#    next_file = os.path.join(
#        folder,
#        f"{prefix}_{next_date.day:02d}_{next_date.month:02d}_{next_date.year}_{suffix}",
#    )
#
#    # ---------------------------------------------------------
#    # Read current raster and mask only
#    # ---------------------------------------------------------
#    mask = lfr.from_gdal(
#        mask_file,
#        partition_shape=partition_shape,
#    )
#
#    current = lfr.from_gdal(
#        current_file,
#        partition_shape=partition_shape,
#    )
#
#    # ---------------------------------------------------------
#    # Determine which cells need interpolation
#    # ---------------------------------------------------------
#    fill_condition = lfr.logical_and(
#        lfr.logical_not(lfr.valid(current)),
#        mask,
#    )
#
#    has_missing = lfr.any(fill_condition)
#
#    if not has_missing:
#        print(
#            f"No missing values requiring interpolation in "
#            f"{current_file}"
#        )
#        return
#
#    print(
#        f"Missing values detected in {current_file}. "
#        f"Reading neighbouring rasters..."
#    )
#
#    # ---------------------------------------------------------
#    # Read neighbouring rasters only when needed
#    # ---------------------------------------------------------
#    if not os.path.exists(prev_file):
#        raise FileNotFoundError(
#            f"Previous-day raster not found:\n{prev_file}"
#        )
#
#    if not os.path.exists(next_file):
#        raise FileNotFoundError(
#            f"Next-day raster not found:\n{next_file}"
#        )
#
#    prev = lfr.from_gdal(
#        prev_file,
#        partition_shape=partition_shape,
#    )
#
#    next = lfr.from_gdal(
#        next_file,
#        partition_shape=partition_shape,
#    )
#
#    # ---------------------------------------------------------
#    # Check whether interpolation is possible
#    # ---------------------------------------------------------
#    invalid_prev_next = lfr.logical_or(
#        lfr.logical_not(lfr.valid(prev)),
#        lfr.logical_not(lfr.valid(next)),
#    )
#
#    problem_cells = lfr.logical_and(
#        fill_condition,
#        invalid_prev_next,
#    )
#
#    has_problem = lfr.any(problem_cells)
#
#    if has_problem:
#        raise ValueError(
#            f"Cannot interpolate: some missing pixels in "
#            f"{current_file} have invalid values in the "
#            f"previous or next day raster."
#        )
#
#    # ---------------------------------------------------------
#    # Temporal interpolation
#    # ---------------------------------------------------------
#    fill_values = (prev + next) / 2.0
#
#    result = lfr.where(
#        fill_condition,
#        fill_values,
#        current,
#    )
#
#    # ---------------------------------------------------------
#    # Save result
#    # ---------------------------------------------------------
#    output_full_path = os.path.join(
#        output_path,
#        base,
#    )
#
#    lfr.to_gdal(
#        result,
#        output_full_path,
#    )
#
#    print(
#        f"Interpolation completed. "
#        f"Output saved to {output_full_path}"
#    )
#    

#import os
#from datetime import datetime, timedelta
#
#import numpy as np
#from osgeo import gdal
#
#
#def nodata_fill_gdal(
#    current_file: str,
#    mask_file: str,
#    output_file: str,
#):
#    print("inside function nodata_fill")
#
#    # ---------------------------------------------------------
#    # Get current date from filename
#    # ---------------------------------------------------------
#
#    folder = os.path.dirname(current_file) or "."
#    base = os.path.basename(current_file)
#
#    parts = base.split("_")
#
#    day = int(parts[2])
#    month = int(parts[3])
#    year = int(parts[4])
#
#    current_date = datetime(year, month, day)
#
#    prefix = "_".join(parts[:2])
#    suffix = "_".join(parts[5:])
#
#    # ---------------------------------------------------------
#    # Previous and next day
#    # ---------------------------------------------------------
#
#    previous_date = current_date - timedelta(days=1)
#    next_date = current_date + timedelta(days=1)
#
#    previous_file = os.path.join(
#        folder,
#        f"{prefix}_{previous_date.day:02d}_"
#        f"{previous_date.month:02d}_"
#        f"{previous_date.year}_{suffix}",
#    )
#
#    next_file = os.path.join(
#        folder,
#        f"{prefix}_{next_date.day:02d}_"
#        f"{next_date.month:02d}_"
#        f"{next_date.year}_{suffix}",
#    )
#
#    # ---------------------------------------------------------
#    # Read rasters
#    # ---------------------------------------------------------
#
#    current_ds = gdal.Open(current_file)
#    previous_ds = gdal.Open(previous_file)
#    next_ds = gdal.Open(next_file)
#    mask_ds = gdal.Open(mask_file)
#
#    current_band = current_ds.GetRasterBand(1)
#    previous_band = previous_ds.GetRasterBand(1)
#    next_band = next_ds.GetRasterBand(1)
#    mask_band = mask_ds.GetRasterBand(1)
#
#    current = current_band.ReadAsArray()
#    previous = previous_band.ReadAsArray()
#    next_day = next_band.ReadAsArray()
#    mask = mask_band.ReadAsArray()
#
#    # ---------------------------------------------------------
#    # Read NoData from current TIFF
#    # ---------------------------------------------------------
#
#    nodata = current_band.GetNoDataValue()
#
#    if nodata is None:
#        raise ValueError(
#            f"NoData value is not defined in {current_file}"
#        )
#
#    # ---------------------------------------------------------
#    # Interpolation
#    # ---------------------------------------------------------
#
#    interpolated_current = (
#        previous.astype(np.float32)
#        + next_day.astype(np.float32)
#    ) / 2.0
#
#    # ---------------------------------------------------------
#    # Fill only:
#    #
#    # current == NoData
#    # AND
#    # mask == 1
#    # ---------------------------------------------------------
#
##    fill_condition = (
##        (current == nodata)
##        & (mask == 1)
##    )
#    
#    fill_condition = (
#        (current == nodata)
#    )
#
#    updated_current = np.where(
#        fill_condition,
#        interpolated_current,
#        current,
#    )
#
#    # ---------------------------------------------------------
#    # Create output directory
#    # ---------------------------------------------------------
#
#    os.makedirs(
#        os.path.dirname(output_file),
#        exist_ok=True,
#    )
#    
#    # ---------------------------------------------------------
#    # Create output as copy of current TIFF
#    # ---------------------------------------------------------
#
#    output_ds = gdal.GetDriverByName("GTiff").CreateCopy(
#        output_file,
#        current_ds,
#        strict=0,
#    )
#
#    output_band = output_ds.GetRasterBand(1)
#
#    output_band.WriteArray(updated_current)
#
#    output_band.FlushCache()
#    output_ds.FlushCache()
#
#    # ---------------------------------------------------------
#    # Close
#    # ---------------------------------------------------------
#
#    output_ds = None
#    current_ds = None
#    previous_ds = None
#    next_ds = None
#    mask_ds = None
#
#    print(f"Output saved to: {output_file}")


import os
from datetime import datetime, timedelta

import numpy as np
import rasterio


def nodata_fill_gdal(
    current_file: str,
    mask_file: str,
    output_file: str,
):
    print("inside function nodata_fill")

    # ---------------------------------------------------------
    # Get current date from filename
    # ---------------------------------------------------------

    folder = os.path.dirname(current_file) or "."
    base = os.path.basename(current_file)

    parts = base.split("_")

    day = int(parts[2])
    month = int(parts[3])
    year = int(parts[4])

    current_date = datetime(year, month, day)

    prefix = "_".join(parts[:2])
    suffix = "_".join(parts[5:])

    # ---------------------------------------------------------
    # Previous and next day
    # ---------------------------------------------------------

    previous_date = current_date - timedelta(days=1)
    next_date = current_date + timedelta(days=1)

    previous_file = os.path.join(
        folder,
        f"{prefix}_{previous_date.day:02d}_"
        f"{previous_date.month:02d}_"
        f"{previous_date.year}_{suffix}",
    )

    next_file = os.path.join(
        folder,
        f"{prefix}_{next_date.day:02d}_"
        f"{next_date.month:02d}_"
        f"{next_date.year}_{suffix}",
    )

    # ---------------------------------------------------------
    # Read current, previous, next
    # ---------------------------------------------------------

    with rasterio.open(current_file) as current_src, \
         rasterio.open(previous_file) as previous_src, \
         rasterio.open(next_file) as next_src:

        current = current_src.read(1)
        previous = previous_src.read(1)
        next_day = next_src.read(1)

        # NoData from current TIFF
        nodata = current_src.nodata

        if nodata is None:
            raise ValueError(
                f"NoData value is not defined in {current_file}"
            )

        print(f"NoData: {nodata}")

        # -----------------------------------------------------
        # Interpolation
        # -----------------------------------------------------

        interpolated_current = (
            previous.astype(np.float32)
            + next_day.astype(np.float32)
        ) / 2.0

        # -----------------------------------------------------
        # Fill ALL current NoData pixels
        # -----------------------------------------------------

        fill_condition = current == nodata

        updated_current = np.where(
            fill_condition,
            interpolated_current,
            current,
        )

        # -----------------------------------------------------
        # COPY COMPLETE PROFILE FROM CURRENT TIFF
        # -----------------------------------------------------

        profile = current_src.profile.copy()

    # ---------------------------------------------------------
    # Write output using the SAME profile
    # ---------------------------------------------------------

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True,
    )

    with rasterio.open(
        output_file,
        "w",
        **profile,
    ) as dst:

        dst.write(
            updated_current,
            1,
        )

    print(f"Output saved to: {output_file}")
    
    

from osgeo import gdal
import os
import numpy as np
import rasterio


def fill_nodata_spatial(
    input_file: str,
    output_file: str,
    max_distance: float = 100,
    smoothing_iterations: int = 0,
):
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")

    # ---------------------------------------------------------
    # Read input raster and its complete profile
    # ---------------------------------------------------------

    with rasterio.open(input_file, "r") as src:

        profile = src.profile.copy()

        current = src.read(1)

        nodata = src.nodata

        if nodata is None:
            raise ValueError(
                f"NoData is not defined in {input_file}"
            )

        print(f"NoData value: {nodata}")
        print(f"Data type: {src.dtypes[0]}")
        print(f"Compression: {src.compression}")
        print(f"CRS: {src.crs}")

    # ---------------------------------------------------------
    # Spatial interpolation using GDAL
    # ---------------------------------------------------------

    # Create a temporary in-memory GDAL raster
    from osgeo import gdal

    driver = gdal.GetDriverByName("MEM")

    mem_ds = driver.Create(
        "",
        current.shape[1],
        current.shape[0],
        1,
        gdal.GDT_Float32,
    )

    mem_band = mem_ds.GetRasterBand(1)

    mem_band.WriteArray(
        current.astype(np.float32)
    )

    mem_band.SetNoDataValue(
        float(nodata)
    )

    # ---------------------------------------------------------
    # Fill NoData spatially
    # ---------------------------------------------------------

    gdal.FillNodata(
        targetBand=mem_band,
        maskBand=None,
        maxSearchDist=max_distance,
        smoothingIterations=smoothing_iterations,
    )

    interpolated = mem_band.ReadAsArray()

    mem_ds = None

    # ---------------------------------------------------------
    # Convert back to original data type
    # ---------------------------------------------------------

    interpolated = interpolated.astype(
        current.dtype
    )

    # ---------------------------------------------------------
    # Write using EXACT INPUT PROFILE
    # ---------------------------------------------------------

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True,
    )

    with rasterio.open(
        output_file,
        "w",
        **profile,
    ) as dst:

        dst.write(
            interpolated,
            1,
        )

    print("Spatial interpolation completed.")
    print(f"Output saved to: {output_file}")
    
    

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt


def fill_nodata_nearest(input_file, output_file):
    """
    Fill all NoData pixels using the value of the nearest valid pixel.

    Original valid pixels are kept unchanged.
    """

    with rasterio.open(input_file) as src:
        data = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is None:
        raise ValueError(
            f"NoData value is not defined: {input_file}"
        )

    # Valid pixels
    valid = data != nodata

    # If there are no NoData pixels, simply copy the raster
    if np.all(valid):
        filled = data.copy()

    else:

        if not np.any(valid):
            raise ValueError(
                f"Cannot fill NoData: no valid pixels in {input_file}"
            )

        # Find nearest valid pixel for every pixel
        _, nearest_indices = distance_transform_edt(
            ~valid,
            return_distances=True,
            return_indices=True,
        )

        # Values of nearest valid pixels
        nearest_values = data[
            nearest_indices[0],
            nearest_indices[1],
        ]

        # Start with original data
        filled = data.copy()

        # Fill only NoData pixels
        filled[~valid] = nearest_values[~valid]

    # Write with the same profile
    with rasterio.open(
        output_file,
        "w",
        **profile,
    ) as dst:
        dst.write(filled, 1)