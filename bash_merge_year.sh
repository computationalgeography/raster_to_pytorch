#!/bin/bash

##module load 2025
##module load Miniforge3/25.3.0-3
##
##source /sw/arch/RHEL9/EB_production/2025/software/Miniforge3/25.3.0-3/etc/profile.d/conda.sh
##conda activate /home/sfaraji1/.conda/envs/pytorch_env


#export MODULEPATH=/depfg/easybuild/foss2024a/modules/all/
#module load Miniconda3/24.7.1-0
#
#source /depfg/easybuild/foss2024a/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
#
#conda activate jupyter_env

#Note************: conda activate jupyter_env *********************

# Input/output directories

folder_name="swe"     #"swe_20260716"

#in_dir="/archive/depfg/compgeo/projecto/results/swe/${folder_name}"
in_dir="/archive/depfg/compgeo/projecto/results_bilinear/${folder_name}"


#in_dir="/archive/depfg/compgeo/projecto/pilot_data/${folder_name}"

#out_dir="/archive/depfg/compgeo/projecto/results_merged/${folder_name}_merged"
out_dir="/archive/depfg/compgeo/projecto/results_bilinear_merged_swe"

num_ranks=16

mkdir -p "$out_dir"

# Years to process
#years=(
#1981
#1982
#)

years=("$@")  # gets all passed yras_list from slurm

for year in "${years[@]}"; do

    echo "=========================="
    echo "Processing year $year"
    echo "=========================="

    for r0 in "$in_dir"/SWE_*_"${year}"_r0.tif; do

        [ -e "$r0" ] || continue

        file=$(basename "$r0")
        base=${file%_r0.tif}

        echo "Merging $base"

        inputs=()
        missing=0

        for ((i=0; i<num_ranks; i++)); do
            f="${in_dir}/${base}_r${i}.tif"
            if [ ! -f "$f" ]; then
                echo "Missing: $f"
                missing=1
                break
            fi
            inputs+=("$f")
        done
        
        [ $missing -eq 1 ] && continue

#        vrt="${in_dir}/${base}.vrt"
        vrt="${out_dir}/${base}.vrt"

        gdalbuildvrt "$vrt" "${inputs[@]}"

        gdal_translate \
            -co COMPRESS=ZSTD \
            "$vrt" \
            "${out_dir}/${base}.tif"

        rm -f "$vrt"

    done

    echo "Finished year $year"

done

echo "All years finished."