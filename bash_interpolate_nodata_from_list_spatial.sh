
#!/bin/bash

#export MODULEPATH=/depfg/easybuild/foss2024a/modules/all/
#module load LUE/0.3.10-foss-2024a

#conda activate jupyter_env (for gdal numpy ...)

PROJECT_ROOT="/scratch/depfg/faraj001/20260218_Oriol_snow_model_LSTM"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

#file_list="/scratch/depfg/faraj001/20260218_Oriol_snow_model_LSTM/missing_inputs_test.csv"
file_list="/scratch/depfg/faraj001/20260218_Oriol_snow_model_LSTM/missing_inputs.csv"
output_path="/archive/depfg/compgeo/projecto/interpolate_nodata_spatial/"
max_distance=100
smoothing_iterations=0      #1  A non-zero value may change the original values of non-missing data pixels.

python script/interpolate_nodata_spatial.py \
    "${file_list}" \
    "${output_path}" \
    "${max_distance}" \
    "${smoothing_iterations}"

