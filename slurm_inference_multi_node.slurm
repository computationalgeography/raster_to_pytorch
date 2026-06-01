#!/bin/bash
#SBATCH --job-name=lstm_inference_multi_node
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=s.farajigargari@uu.nl
#SBATCH --mail-type=ALL
#SBATCH --partition=fat_genoa
#SBATCH --mem=800G #500G  #250G       # define allocated memory 
#SBATCH --time=02:00:00
#SBATCH -o job_%x_%j.out
#SBATCH -e job_%x_%j.err

#SBATCH --exclusive   
# #SBATCH --constraint=fcn108|fcn109|fcn110|fcn111
#SBATCH --nodelist=fcn108,fcn109,fcn110,fcn111
   


module load 2025
module load Miniforge3/25.3.0-3

source /sw/arch/RHEL9/EB_production/2025/software/Miniforge3/25.3.0-3/etc/profile.d/conda.sh
conda activate /home/sfaraji1/.conda/envs/pytorch_env

export PYTHONPATH="/projects/prjs1930/saeb/20260218_Oriol_snow_model_LSTM:$PYTHONPATH"

srun python script/inference_multi_node.py