#!/bin/bash

#export MODULEPATH=/depfg/easybuild/foss2024a/modules/all/
#module load LUE/0.3.10-foss-2024a

#conda activate jupyter_env  (for rasterio)

PROJECT_ROOT="/scratch/depfg/faraj001/20260218_Oriol_snow_model_LSTM"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

file_list="/scratch/depfg/faraj001/20260218_Oriol_snow_model_LSTM/missing_inputs_test.csv"
mask_file="/archive/depfg/compgeo/projecto/data/inference_mask.tif"
output_path="/archive/depfg/compgeo/projecto/interpolate_nodata/"
partition_size=1000

python script/interpolate_nodata_non_consecutive.py \
    "${file_list}" \
    "${mask_file}" \
    "${output_path}" \
    "${partition_size}"