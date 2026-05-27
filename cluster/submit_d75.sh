#!/bin/bash
#SBATCH --job-name=dgm_d75
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=scaling_d75_%j.out
#SBATCH --error=scaling_d75_%j.err
#SBATCH --partition=gpu

# Go to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# Initialize your local Miniconda and activate the environment
source $HOME/miniconda3/bin/activate
conda activate dgm_env

echo "=========================================="
echo "Scaling Study: d=75"
echo "=========================================="

python experiments/run_scaling.py --dims 75 --n_steps 150000 --lbfgs_steps 100
python experiments/run_comparison.py --dims 75

echo "Scaling & Evaluation d=75 Complete!"
