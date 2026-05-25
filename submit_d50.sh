#!/bin/bash
#SBATCH --job-name=dgm_d50
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=scaling_d50_%j.out
#SBATCH --error=scaling_d50_%j.err
#SBATCH --partition=gpu

# Go to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# Initialize your local Miniconda and activate the environment
source $HOME/miniconda3/bin/activate
conda activate dgm_env

echo "=========================================="
echo "Scaling Study: d=50"
echo "=========================================="

python experiments/run_scaling.py --dims 50 --n_steps 100000 --lbfgs_steps 100

echo "Scaling d=50 Complete!"
