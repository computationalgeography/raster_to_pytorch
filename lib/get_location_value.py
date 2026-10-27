import rasterio
from rasterio.transform import rowcol
#from pyproj import Transformer
    
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt


#def get_pixel_value(tif_path, lon, lat, band=1):
#    """
#    Get raster pixel value from lon/lat.
#
#    Parameters:
#    -----------
#    tif_path : str
#        Path to GeoTIFF
#    lon : float
#        Longitude
#    lat : float
#        Latitude
#    band : int
#        Raster band (default=1)
#
#    Returns:
#    --------
#    float
#        Pixel value at nearest location
#    """
#
#    with rasterio.open(tif_path) as src:
#
#        # Step 1: transform WGS84 (lon/lat) ? raster CRS
#        if src.crs.to_string() != "EPSG:4326":
#            transformer = Transformer.from_crs(
#                "EPSG:4326", src.crs, always_xy=True
#            )
#            x, y = transformer.transform(lon, lat)
#        else:
#            x, y = lon, lat
#
#        # Step 2: convert coordinates ? pixel index
#        row, col = rowcol(src.transform, x, y)
#
#        # Step 3: safety check
#        if (
#            row < 0 or col < 0 or
#            row >= src.height or col >= src.width
#        ):
#            raise ValueError("Point is outside raster bounds")
#
#        # Step 4: read pixel value (nearest neighbor)
#        value = src.read(band)[row, col]
#        
#        print("value, row, col: ", value, row, col)
#
#        return value, row, col
        

def get_pixel_value(tif_path, lon, lat):
    with rasterio.open(tif_path) as src:

        # bounds check
        minx, miny, maxx, maxy = src.bounds
        if not (minx <= lon <= maxx and miny <= lat <= maxy):
            print("Outside raster bounds")
            return None, None, None

        # pixel coordinates (row, col)
        row, col = rowcol(src.transform, lon, lat)

        val = list(src.sample([(lon, lat)]))[0][0]

        if val == src.nodata:
            print("NoData at location")
            return None, row, col

        print(f"value={val}, row={row}, col={col}")
        return float(val), row, col


def plot_timeseries(df, station):

    plt.figure(figsize=(10, 6))

    df["swe"].plot(
        label=f"Station {station}"
    )

    plt.xlabel("Date")
    plt.ylabel("SWE (mm)")
    plt.ylim(bottom=0)
    plt.title(f"SWE time series for station {station}")
    plt.legend()

    outfile = f"swe_timeseries_station_{station}.png"

    plt.savefig(outfile, dpi=300)
    plt.close()

    print(f"Saved plot: {outfile}")


#def extract_swe_timeseries(folder, lon, lat):
#
##    files = sorted(glob.glob(os.path.join(folder, "SWE_*_r*.tif")))
#    files = sorted(glob.glob(os.path.join(folder, "SWE_*.tif")))
#
#    dates = []
#    values = []
#
#    for file in files:
#        try:
#            value, row, col = get_pixel_value(file, lon, lat)
#
##            print(
##                f"{os.path.basename(file)} "
##                f"date={os.path.basename(file)} "
##                f"value={value} row={row} col={col}"
##            )
#
#            # Extract date: SWE_DD_MM_YYYY_r4.tif
##            match = re.search(
##                r"SWE_(\d{2}_\d{2}_\d{4})_r\d+\.tif",
##                os.path.basename(file)
##            )
#
#            match = re.search(
#                r"SWE_(\d{2}_\d{2}_\d{4})\.tif$",
#                os.path.basename(file)
#            )
#
#
#            if match:
#                date = pd.to_datetime(
#                    match.group(1),
#                    format="%d_%m_%Y"
#                )
#
#                dates.append(date)
#                values.append(value)
#
#        except Exception as e:
#            print(f"Skipping {file}: {e}")
#
#    df = pd.DataFrame(
#        {
#            "date": dates,
#            "swe": values,
#        }
#    )
#
#    df = df.sort_values("date")
#    df = df.set_index("date")
#
#    return df
    
    
def extract_swe_timeseries(folder, lon, lat, start_date=None, end_date=None):

  files = sorted(glob.glob(os.path.join(folder, "SWE_*.tif")))

  dates = []
  values = []

  if start_date is not None:
      start_date = pd.to_datetime(start_date)

  if end_date is not None:
      end_date = pd.to_datetime(end_date)

  for file in files:

      filename = os.path.basename(file)

      match = re.search(
          r"SWE_(\d{2}_\d{2}_\d{4})\.tif$",
          filename
      )

      if not match:
          continue

      date = pd.to_datetime(
          match.group(1),
          format="%d_%m_%Y"
      )

      # Skip files outside requested period
      if start_date is not None and date < start_date:
          continue

      if end_date is not None and date > end_date:
          continue

      try:
          value, row, col = get_pixel_value(file, lon, lat)

          dates.append(date)
          values.append(value)

      except Exception as e:
          print(f"Skipping {file}: {e}")

  df = pd.DataFrame(
      {
          "date": dates,
          "swe": values,
      }
  )

  df = df.sort_values("date")
  df = df.set_index("date")

  return df  