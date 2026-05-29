#!/bin/bash
#SBATCH --job-name=update_hmc
#SBATCH --partition=gpu
#SBATCH --account=iitr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=update_hmc_%j.out
#SBATCH --error=update_hmc_%j.err

echo "Loading conda environment..."
source /home/siddhartha_g_ma.iitr/miniconda3/etc/profile.d/conda.sh
conda activate dgm_env

cd /home/siddhartha_g_ma.iitr/dgm_option_pricing

echo "Starting Hybrid MC update..."
python experiments/update_hmc.py

echo "Done! Updating git..."
git add results/publication/comparison_summary.json
git commit -m "Update Hybrid MC for d<=10 with unbiased Delta control variate"
git push origin main
