#!/bin/bash
# Submit script for d=50 scaling run

# Ensure conda environment is active (adjust if needed on your cluster)
# source activate base

# Run the scaling experiment for d=50
# We reduce n_steps to 100,000 to ensure it finishes within the 24 hour deadline
python experiments/run_scaling.py --dims 50 --n_steps 100000 --lbfgs_steps 100
