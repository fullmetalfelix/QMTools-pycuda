#!/bin/bash
#SBATCH --gres=gpu:4        # Request GPUs
#SBATCH --constraint=a100   # Request specific GPU architecture
#SBATCH --time=02-00:00:00  # Job time allocation
#SBATCH --mem=32G           # Memory
#SBATCH -c 6                # Number of cores
#SBATCH -J grid_gnn         # Job name
#SBATCH -o grid_gnn.log     # Output file

# Load modules
module load mamba
source activate qmtools2

# Print job info
echo "Job ID: "$SLURM_JOB_ID
echo "Job Name: "$SLURM_JOB_NAME

# Print environment info
which python
python --version
conda info --envs
conda list
pip list

num_gpus=$(echo "$SLURM_JOB_GPUS" | sed -e $'s/,/\\\n/g' | wc -l)
echo "Number of GPUs: $num_gpus"

# Run script
torchrun \
    --standalone \
    --nnodes 1 \
    --nproc_per_node $num_gpus \
    --max_restarts 0 \
    train_grid_gnn.py