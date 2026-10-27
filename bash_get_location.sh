#!/bin/bash

PROJECT_ROOT="/projects/prjs1930/saeb/20260218_Oriol_snow_model_LSTM"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"


file_path="../../pilot_data/DEM/elevation_mn.tif"
#file_path="../../pilot_data/DEM/slope_mn.tif"
#file_path="../../pilot_data/DEM/northness_mn.tif"
#file_path="../../pilot_data/DEM/tpi_mn.tif"
#file_path="../../pilot_data/DEM/eastness_mn.tif"
#file_path="../../pilot_data/swe/SWE_14_10_1980_r4.tif"

#eastness_mn.tif  elevation_1KMsd_GMTEDmd.tif  elevation_mn.tif  northness_mn.tif  slope_mn.tif  tpi_mn.tif

# Station coordinates

station_10012_lon=41.7667
station_10012_lat=54.9500

station_30529_lon=7.8488
station_30529_lat=45.8522

#station_30529=(lat=45.8522, lon=7.8488)

python script/get_location_value.py "$file_path" "$station_10012_lon" "$station_10012_lat"
python script/get_location_value.py "$file_path" "$station_30529_lon" "$station_30529_lat"

#python script/inference_with_mask.py

#python -m memory_profiler script/inference.py > memory_profile_inference.txt

#python -m scalene --outfile profile_cpu_mem.txt script/inference_multi_node_hybrid_model.py