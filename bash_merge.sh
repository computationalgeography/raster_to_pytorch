module load 2025
module load Miniforge3/25.3.0-3

source /sw/arch/RHEL9/EB_production/2025/software/Miniforge3/25.3.0-3/etc/profile.d/conda.sh
conda activate /home/sfaraji1/.conda/envs/pytorch_env

in_dir="../../pilot_data/swe"
out_dir="../../pilot_data/swe_merged"


mkdir -p "$out_dir"

for r0 in $in_dir/*_r0.tif; do
    file=$(basename "$r0")
    base=${file%_r0.tif}

    gdalbuildvrt "${in_dir}/${base}.vrt" \
        "${in_dir}/${base}_r0.tif" \
        "${in_dir}/${base}_r1.tif" \
        "${in_dir}/${base}_r2.tif" \
        "${in_dir}/${base}_r3.tif"

    gdal_translate \
        -co COMPRESS=ZSTD \
        "${in_dir}/${base}.vrt" \
        "${out_dir}/${base}.tif"

    rm "${in_dir}/${base}.vrt"
done
