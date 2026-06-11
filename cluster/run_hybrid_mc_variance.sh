#!/bin/bash
#SBATCH --job-name=hmc_variance
#SBATCH --partition=gpu
#SBATCH --account=iitr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=hmc_variance_%j.out
#SBATCH --error=hmc_variance_%j.err

echo "Loading conda environment..."
source /home/siddhartha_g_ma.iitr/miniconda3/etc/profile.d/conda.sh
conda activate dgm_env

cd /home/siddhartha_g_ma.iitr/dgm_option_pricing

echo "=========================================="
echo "Computing Hybrid MC Variance Reduction for d = 7, 10, 25, 50, 100"
echo "=========================================="

python -u experiments/run_hybrid_mc.py --dims 7 10 25 50 100 --n_mc 100000 --force

echo "Done! Updating git..."
git add results/hybrid_mc/*/*.json results/hybrid_mc/hybrid_mc_summary.json
git commit -m "Update Hybrid MC variance reduction results for d=7,10,25,50,100"
git push origin main
