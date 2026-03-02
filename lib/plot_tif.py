#!/usr/bin/env python3
import sys
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np

def show_tif(tif_file, array_small_num=10):
    # Open the TIFF file
    dataset = gdal.Open(tif_file)
    if dataset is None:
        print(f"Error: Could not open {tif_file}")
        return

    # Print basic information
    print(f"File: {tif_file}")
    print(f"Raster size: {dataset.RasterXSize} x {dataset.RasterYSize} (cols x rows)")
    print(f"Number of bands: {dataset.RasterCount}")
    print(f"Projection: {dataset.GetProjection()}")
    gt = dataset.GetGeoTransform()
    print(f"Geotransform: {gt}")
    print(f"Data type of first band: {gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType)}")

    # Read the first band
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()

    # --- Optional: downsample for plotting (skip every N pixels) ---
    array_small = array[::array_small_num, ::array_small_num]

    # --- Plot row/column indices ---
    plt.figure(figsize=(10, 5))
    plt.imshow(array_small, cmap='terrain')
    plt.colorbar(label='Value')
    plt.title(f"Visualization of {tif_file} (pixel indices)")
    plt.xlabel("X (column index)")
    plt.ylabel("Y (row index)")
    plt.show()

    # --- Plot with real coordinates (longitude/latitude) ---
    nrows, ncols = array.shape
    x = gt[0] + np.arange(ncols) * gt[1]   # longitude
    y = gt[3] + np.arange(nrows) * gt[5]   # latitude (note: gt[5] is negative)

    plt.figure(figsize=(10, 5))
    plt.imshow(array_small, cmap='terrain', extent=[x[0], x[-1], y[-1], y[0]])
    plt.colorbar(label='Value')
    plt.xlabel("Longitude (°)")
    plt.ylabel("Latitude (°)")
    plt.title(f"Visualization of {tif_file} (degrees)")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python show_tif.py <file.tif>")
        sys.exit(1)

    tif_file = sys.argv[1]
    show_tif(tif_file)
