import calendar
import pathlib
import shutil
import tempfile

import netCDF4
import numpy as np
import pandas as pd
from osgeo import gdal

gdal.UseExceptions()


def convert(args):
    """Creates monthly NetCDF4 datasets"""

    month = args[0]
    year = args[1]
    input_dir = args[2]
    output_dir = args[3]

    target_filename = f"HySWE_{month:02d}_{year}.nc"
    target_path = pathlib.Path(output_dir, target_filename)

    if target_path.exists():
        return

    with tempfile.TemporaryDirectory(
        dir=pathlib.Path.cwd(), prefix="tmp_", delete=False
    ) as tmp_dir:
        # Open first dataset to read some attributes
        output_path = pathlib.Path(tmp_dir, target_filename)

        ds = gdal.Open(pathlib.Path(input_dir, f"SWE_01_{month:02d}_{year}.tif"))
        assert ds

        array = ds.ReadAsArray()
        nlat, nlon = np.shape(array)

        transform = ds.GetGeoTransform()
        band = ds.GetRasterBand(1)
        no_data = int(band.GetNoDataValue())

        # Create arrays with Latitude/Longitude values
        half_lon = transform[1] / 2.0
        half_lat = transform[5] / 2.0

        lon_val = np.arange(nlon) * transform[1] + transform[0] + half_lon
        lat_val = np.arange(nlat) * transform[5] + transform[3] + half_lat

        dataset = netCDF4.Dataset(output_path, "w", data_model="NETCDF4", clobber=True)

        # Chunking is optional, but can improve access a lot:
        # (see: http://www.unidata.ucar.edu/blogs/developer/entry/chunking_data_choosing_shapes)
        chunk_lon = 1000
        chunk_lat = 1000
        chunk_time = 1

        # Timesteps this month
        nr_days = calendar.monthrange(year, month)[1]

        # Create dimensions, variables and attributes:
        dataset.createDimension("lon", nlon)
        dataset.createDimension("lat", nlat)
        dataset.createDimension("time", nr_days)

        time = dataset.createVariable("time", "f4", ("time"))
        time.units = f"days since {year}-{month:02d}-01 00:00:00"
        time.standard_name = "time"
        time.calendar = "standard"
        time.axis = "T"

        lon = dataset.createVariable("lon", "f8", ("lon"))
        lon.units = "degrees_east"
        lon.standard_name = "longitude"
        lon.long_name = "longitude"
        lon.axis = "X"

        lat = dataset.createVariable("lat", "f8", ("lat"))
        lat.units = "degrees_north"
        lat.standard_name = "latitude"
        lat.long_name = "latitude"
        lat.axis = "Y"

        # Create container variable for CRS: lon/lat WGS84 datum
        crs = dataset.createVariable("crs", "f4")
        crs.long_name = "CRS definition"
        crs.grid_mapping_name = "latitude_longitude"
        crs.longitude_of_prime_meridian = 0.0
        crs.semi_major_axis = 6378137.0
        crs.inverse_flattening = 298.257223563
        crs.crs_wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
        crs.proj_code = "EPSG:4326"

        # Compression level
        level = 9

        # Create float variable for precipitation data, with chunking
        ssa = dataset.createVariable(
            "ssa",
            "u2",
            ("time", "lat", "lon"),
            compression="zlib",
            complevel=level,
            chunksizes=[chunk_time, chunk_lat, chunk_lon],
            fill_value=no_data,
        )
        ssa.units = "kg m-2"
        ssa.standard_name = "surface snow amount"
        ssa.long_name = "Snow water equivalent"
        ssa.grid_mapping = "crs"
        ssa.set_auto_maskandscale(False)

        # Additional attributes
        dataset.Conventions = "CF-1.6"
        dataset.title = "HySWE"
        dataset.version = "v1.0"
        dataset.comments = "The values in the latitude and longitude variables represent the grid cell centres."
        dataset.date_created = "2026-10-08"
        dataset.creator_name = "Oriol Pomarol Moya"
        dataset.creator_email = "o.pomarolmoya@uu.nl"
        dataset.institution = "Utrecht University"
        dataset.summary = "Daily 1-km northern hemisphere snow water equivalent reconstruction for 1980–2020 based on hybrid machine learning."
        dataset.license = "CC-BY-4.0"

        dataset.time_coverage_start = f"{year}-{month:02d}-01 00:00:00"
        dataset.time_coverage_end = f"{year}-{month:02d}-{nr_days:02d} 00:00:00"

        dataset.valid_min = 0.0
        dataset.units = "kg m-2"

        # Write Latitude/Longitude
        lon[:] = lon_val
        lat[:] = lat_val

        for day in range(nr_days):
            time[day] = day
            ssa_path = pathlib.Path(
                input_dir, f"SWE_{(day+1):02d}_{month:02d}_{year}.tif"
            )
            ssa = gdal.Open(ssa_path)
            ssa[day, :, :] = ssa.ReadAsArray()
            ssa.valid_min = 0
            ssa.Close()

        dataset.close()

        shutil.move(output_path, target_path)
        print(f"generated {target_path}", flush=True)


if __name__ == "__main__":

    import argparse
    from concurrent.futures import ProcessPoolExecutor

    parser = argparse.ArgumentParser(
        description="Simple helper script creating monthly NetCDF4"
    )
    parser.add_argument("source", type=str, help="Source path to input files")
    parser.add_argument("destination", type=str, help="Output path")
    parser.add_argument("cpu", type=int, help="Number of CPUs to use")
    args = parser.parse_args()

    input_dir = args.source
    output_dir = args.destination
    cpus = args.cpu

    # Months within the simulation period
    pr = pd.period_range(start="1980-10", end="2020-09", freq="M")

    # Create arguments per task
    tasks = tuple([(period.month, period.year, input_dir, output_dir) for period in pr])

    with ProcessPoolExecutor(max_workers=cpus) as executor:
        executor.map(convert, tasks, chunksize=1)
