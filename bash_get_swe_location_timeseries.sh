#!/bin/bash

module load 2025
module load Miniforge3/25.3.0-3

source /sw/arch/RHEL9/EB_production/2025/software/Miniforge3/25.3.0-3/etc/profile.d/conda.sh
conda activate /home/sfaraji1/.conda/envs/pytorch_env


PROJECT_ROOT="/projects/prjs1930/inference/20260218_Oriol_snow_model_LSTM"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"


#swe_folder="../../pilot_data/20260614_swe_oriol"
#swe_folder="../../pilot_data/swe"
swe_folder="../../pilot_data/swe_6_years_merged"


# Station coordinates

station_10012_lon=41.7667
station_10012_lat=54.9500

station_30529_lon=7.8488
station_30529_lat=45.8522


python script/get_swe_location_timeseries.py "$swe_folder" "$station_30529_lon" "$station_30529_lat" "station_30529"
python script/get_swe_location_timeseries.py "$swe_folder" "$station_10012_lon" "$station_10012_lat" "station_10012"

#python script/inference_with_mask.py

#python -m memory_profiler script/inference.py > memory_profile_inference.txt

#python -m scalene --outfile profile_cpu_mem.txt script/inference_multi_node_hybrid_model.py