#!/bin/bash
#SBATCH --job-name=test_from_checkpoint
# #SBATCH --array=0-15      #0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16      #32    #     16     #32          #16
#SBATCH --mail-user=o.pomarolmoya@uu.nl
#SBATCH --mail-type=END # ,ARRAY_TASKS
#SBATCH --partition=fat_rome     #     fat_genoa
# #SBATCH --mem=250G     #800G #500G  #250G       # define allocated memory 
#SBATCH --time=01:00:00       #12:00:00
#SBATCH -o job_%x_%j.out
#SBATCH -e job_%x_%j.err

# # SBATCH --exclusive   (remove this it may takes all cores in node)
# #SBATCH --constraint=fcn108|fcn109|fcn110|fcn111
# #SBATCH --nodelist=fcn108,fcn109,fcn110,fcn111
   


module load 2025
module load Miniforge3/25.3.0-3

source /sw/arch/RHEL9/EB_production/2025/software/Miniforge3/25.3.0-3/etc/profile.d/conda.sh
conda activate /home/opomarolmoya1/.conda/envs/pytorch_env


export PYTHONPATH="/projects/prjs1930/saeb/20260218_Oriol_snow_model_LSTM:$PYTHONPATH"

echo "========================================"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "========================================"

#srun python script/inference_multi_node.py
#srun python script/inference_multi_node_hybrid_model.py
srun python script/inference_multi_node_hybrid_model_save_states.py
