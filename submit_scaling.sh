#!/bin/bash
#SBATCH --job-name=dgm_scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=scaling_%j.out
#SBATCH --error=scaling_%j.err
#SBATCH --partition=gpu

# Go to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# Initialize your local Miniconda and activate the environment
source $HOME/miniconda3/bin/activate
conda activate dgm_env

echo "Starting Scaling Study for d=7, 10"
python experiments/run_scaling.py --dims 7 10
echo "Scaling Study Complete!"
