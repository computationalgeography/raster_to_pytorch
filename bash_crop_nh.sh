
#!/bin/bash

PROJECT_ROOT="/scratch/depfg/faraj001/20260218_Oriol_snow_model_LSTM"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

in_file_path="/archive/depfg/data/chelsa"
#in_file_path="/scratch/depfg/faraj001/20260218_Oriol_snow_model_LSTM"
out_file_path="/archive/depfg/compgeo/projecto/data"

output_data_type="Float32"

folder_name="pr"    #"tas"    #"rsds"

in_dir="${in_file_path}/${folder_name}"
out_dir="${out_file_path}/${folder_name}"

# create output directory if it does not exist
mkdir -p "$out_dir"

#compress_method="ZSTD"

for tif_file in "$in_dir"/*.tif
do
    filename=$(basename "$tif_file")
    out_file="${out_dir}/${filename}"

#    python script/crop_nh.py "$tif_file" "$out_file" "$output_data_type" compress="$compress_method"
    python script/crop_nh.py "$tif_file" "$out_file" "$output_data_type"
done