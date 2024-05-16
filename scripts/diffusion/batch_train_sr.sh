#!/bin/bash
#SBATCH --gres=gpu:1        # Request GPUs
#SBATCH --constraint=a100   # Request specific GPU architecture
#SBATCH --time=01-00:00:00  # Job time allocation
#SBATCH --mem=16G           # Memory
#SBATCH -c 4                # Number of cores
#SBATCH -J train_sr        # Job name
#SBATCH -o train_sr.log    # Output file

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
python train_sr.py