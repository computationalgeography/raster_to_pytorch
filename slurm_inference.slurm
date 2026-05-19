#!/bin/bash
#SBATCH --job-name=lstm_inference
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16     #12     #8    # define number of cores 
#SBATCH --nodes=1

# #SBATCH --ntasks-per-node=1
# #SBATCH --threads-per-core=1     # define number of threads per core it can be 1 or 2 (if we have duale core)

# #SBATCH --partition=fat_rome
# #SBATCH --nodelist=fcn12     #fcn10

#SBATCH --partition=fat_genoa
#SBATCH --nodelist=fcn104 


# #SBATCH --gres=gpu
#SBATCH --mem=800G   #500G  #250G       # define allocated memory 
#SBATCH -o job_%x_%j.out  
#SBATCH -e job_%x_%j.err
#SBATCH --mail-user=s.farajigargari@uu.nl
#SBATCH --mail-type=ALL

#SBATCH --time=02:00:00    # run time

# module purge


# load conda
module load 2025

module load Miniforge3/25.3.0-3


# activate environment
#source $(conda info --base)/etc/profile.d/conda.sh
#conda activate jupyter_env

# initialize conda
source /sw/arch/RHEL9/EB_production/2025/software/Miniforge3/25.3.0-3/etc/profile.d/conda.sh

# activate environment (robust path-based)
conda activate /home/sfaraji1/.conda/envs/pytorch_env

# sanity checks
echo "CONDA ENV:"
echo $CONDA_DEFAULT_ENV

bash bash_inference.sh