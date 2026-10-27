#!/bin/bash

PROJECT_ROOT="/scratch/depfg/faraj001/20260218_Oriol_snow_model_LSTM"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

in_file_path="/archive/depfg/data/chelsa"
out_file_path="/scratch/depfg/faraj001/20260218_Oriol_snow_model_LSTM/missing_date_list"
#out_file_path="/archive/depfg/compgeo/projecto/data"


folder_name="rsds"      #"pr"    #"tas"    #"rsds"

in_dir="${in_file_path}/${folder_name}"
out_dir="${out_file_path}/${folder_name}"

# create output directory if it does not exist
mkdir -p "$out_dir"

python script/find_missing_day.py "$in_dir" "$out_dir"
