#!/bin/bash
#SBATCH --job-name=dgm_comparison
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=comparison_%j.out
#SBATCH --error=comparison_%j.err
#SBATCH --partition=gpu
#SBATCH --dependency=singleton

# Go to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# Initialize your local Miniconda and activate the environment
source $HOME/miniconda3/bin/activate
conda activate dgm_env

echo "Starting Comparison for d=2, 3, 5, 7, 10"
python experiments/run_comparison.py --dims 2 3 5 7 10
echo "Comparison Complete!"
