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

echo "=========================================="
echo "Full Scaling Study: d=1, 2, 3, 5, 7, 10"
echo "=========================================="
echo ""
echo "How this works:"
echo "  - Completed dimensions (valid result.json) are SKIPPED"
echo "  - Incomplete dimensions AUTO-RESUME from last checkpoint"
echo "  - If job times out at 24h, just resubmit: sbatch submit_scaling.sh"
echo ""

# On first run only: delete old stale result.json files so nothing is wrongly skipped.
# Checkpoints (best_model.pt, latest_model.pt) are KEPT for auto-resume.
# After first run, comment out or remove these lines:
# find results/scaling -name "result.json" -delete 2>/dev/null

python experiments/run_scaling.py --dims 1 2 3 5 7 10

echo "Scaling Study Complete!"
