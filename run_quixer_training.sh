#!/bin/bash
#
# Run Quixer Genomics Training
# Uses the quixer conda environment (Python 3.11+)
#

# Set Python from quixer environment
PYTHON=/scratch/cbjp404/conda_envs/quixer/bin/python
PIP=/scratch/cbjp404/conda_envs/quixer/bin/pip

# Install Quixer package if not already installed
cd /scratch/cbjp404/bradford_hackathon_2025/Quixer
$PIP install -e . --quiet 2>/dev/null || true

# Run training
cd /scratch/cbjp404/bradford_hackathon_2025
$PYTHON train_quixer_genomics.py "$@"
