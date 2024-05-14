#!/bin/bash
#SBATCH --gres=gpu:1        # Request GPUs
#SBATCH --constraint=a100   # Request specific GPU architecture
#SBATCH --time=01-12:00:00  # Job time allocation
#SBATCH --mem=16G           # Memory
#SBATCH -c 4                # Number of cores
#SBATCH -J gen_densities    # Job name
#SBATCH -o gen_%a.log       # Output file %a = array id
#SBATCH --array=0,1,2,3     # Divide work over 4 different jobs

# Load modules
module load mamba
source activate qmtools

# Print job info
echo "Job ID: "$SLURM_JOB_ID
echo "Job Name: "$SLURM_JOB_NAME

# Print environment info
which python
python --version
conda info --envs
conda list
pip list

# Run script
python generate_density_database.py 4 $SLURM_ARRAY_TASK_ID
