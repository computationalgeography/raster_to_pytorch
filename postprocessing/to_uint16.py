import calendar
import pathlib
import shutil
import tempfile

import numpy as np
import pandas as pd
from osgeo import gdal

gdal.UseExceptions()


def convert(args):
    """Converts all datasets of given month"""

    month = args[0]
    year = args[1]
    input_dir = args[2]
    output_dir = args[3]

    nr_days = calendar.monthrange(year, month)[1]

    for timestep in range(1, nr_days + 1):

        filename = f"SWE_{timestep:02d}_{month:02d}_{year}.tif"
        destination_path = pathlib.Path(output_dir, filename)

        if pathlib.Path(destination_path).exists():
            continue

        # Options for the output datasets
        metadata_options = {}
        creation_options = ["COMPRESS=DEFLATE", "PREDICTOR=2", "ZLEVEL=9"]
        translate_options = gdal.TranslateOptions(
            format="GTiff",
            strict=True,
            stats=True,
            outputType=gdal.GDT_UInt16,
            creationOptions=creation_options,
            metadataOptions=metadata_options,
        )

        with tempfile.TemporaryDirectory(
            dir=pathlib.Path.cwd(),
            prefix="tmp_",
            delete=True,
            ignore_cleanup_errors=True,
        ) as tmp_dir:
            source_name = pathlib.Path(input_dir, filename)
            in_ds = gdal.Open(source_name, gdal.GA_ReadOnly)
            tmp_tif_path = pathlib.Path(tmp_dir, "tmp.tif")

            # First, reassign the nodata value on source dataatype
            with gdal.ExceptionMgr(useExceptions=True):

                out_ds = gdal.GetDriverByName("GTiff").CreateCopy(
                    tmp_tif_path,
                    in_ds,
                    strict=True,
                )

                arr = out_ds.ReadAsArray().astype(np.float32)

                assert np.nanmin(arr) < 0.0, f"{source_name} with values smaller than 0"

                data_max = np.nanmax(arr)

                # Replace the previous nodata value with one valid for uint16
                new_nodata = 65535

                assert (
                    data_max < new_nodata
                ), f"{source_name} with values larger than {new_nodata}"

                arr = np.where(arr >= 0.0, arr, new_nodata)
                out_ds.GetRasterBand(1).WriteArray(arr)
                out_ds.GetRasterBand(1).SetNoDataValue(new_nodata)

            # Write the adjusted dataset as uint16

            tmp_u16_path = pathlib.Path(tmp_dir, "u16.tif")

            with gdal.ExceptionMgr(useExceptions=True):

                gdal.Translate(
                    tmp_u16_path,
                    out_ds,
                    options=translate_options,
                )

            print(f"generated {destination_path}", flush=True)
            shutil.move(tmp_u16_path, destination_path)


if __name__ == "__main__":

    import argparse
    from concurrent.futures import ProcessPoolExecutor

    parser = argparse.ArgumentParser(
        description="Simple helper script converting GeoTIFFs to uint16"
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
