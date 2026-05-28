#!/bin/bash
#SBATCH --job-name=dgm_comp_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=comparison_%j.out
#SBATCH --error=comparison_%j.err
#SBATCH --partition=gpu

# Go to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# Initialize your local Miniconda and activate the environment
source $HOME/miniconda3/bin/activate
conda activate dgm_env

echo "Starting Full 5-Way Comparison for all d"
# This will load the pre-trained DGM models from results/scaling, 
# then compute Zhou, Hybrid MC, Vanilla MC, and Deep BSDE (ATM).
python experiments/run_comparison.py --dims 1 2 3 5 7 10 25 50 100 --run_fbsde
echo "Comparison Complete! Results saved to results/publication/comparison_summary.json"
