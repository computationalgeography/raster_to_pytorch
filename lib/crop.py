from osgeo import gdal

def crop_northern_hemisphere(in_file: str, out_file: str, compress: str = "LZW"):
    """
    Crop a raster to the Northern Hemisphere (latitude >= 0)
    and save it with specified compression.

    Parameters
    ----------
    in_file : str
        Path to the input raster file (GeoTIFF, e.g., .tif) that you want to crop.
    out_file : str
        Path where the cropped raster will be saved (GeoTIFF, e.g., .tif).
    compress : str, optional
        Compression type for the output file. Options include "LZW", "DEFLATE", "PACKBITS", etc.
        Default is "LZW".
    """
    ds = gdal.Open(in_file)
    if ds is None:
        raise FileNotFoundError(f"Cannot open {in_file}")

    gt = ds.GetGeoTransform()
    x_min, pixel_width, _, y_max, _, pixel_height = gt
    n_cols, n_rows = ds.RasterXSize, ds.RasterYSize

    xmin = x_min
    xmax = x_min + n_cols * pixel_width
    ymax = y_max
    ymin = 0

    gdal.Translate(
        out_file,
        in_file,
        projWin=[xmin, ymax, xmax, ymin],
        creationOptions=[f"COMPRESS={compress.upper()}", "TILED=YES"]
    )