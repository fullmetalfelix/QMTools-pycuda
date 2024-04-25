#!/bin/bash
#SBATCH --gres=gpu:1        # Request GPUs
#SBATCH --constraint=a100   # Request specific GPU architecture
#SBATCH --time=00-04:00:00  # Job time allocation
#SBATCH --mem=64G           # Memory
#SBATCH -c 4                # Number of cores
#SBATCH -J train_automaton  # Job name
#SBATCH -o fit.log          # Output file

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
python train_automaton_pt.py
