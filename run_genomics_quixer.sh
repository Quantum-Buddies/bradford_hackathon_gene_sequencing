#!/bin/bash
#SBATCH --job-name=genomics_quixer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/cbjp404/bradford_hackathon_2025/logs/quixer_%j.out
#SBATCH --error=/scratch/cbjp404/bradford_hackathon_2025/logs/quixer_%j.err

# Genomics Quixer Training Pipeline
# Runs lambeq-encoded genomic classification on L40s GPUs

echo "========================================="
echo "GENOMICS QUIXER TRAINING PIPELINE"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "========================================="

# Create log directory
mkdir -p /scratch/cbjp404/bradford_hackathon_2025/logs

# Activate conda environment
source ~/.bashrc
conda activate quixer

# Verify GPU access
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Move to Quixer directory
cd /scratch/cbjp404/Quixer

# Stage 1: Data Preprocessing (if not already done)
echo "========================================="
echo "STAGE 1: DATA PREPROCESSING"
echo "========================================="

if [ ! -d "/scratch/cbjp404/bradford_hackathon_2025/processed_data/train" ]; then
    echo "Running preprocessing..."
    python /scratch/cbjp404/bradford_hackathon_2025/preprocess_genomics.py
else
    echo "Preprocessed data already exists, skipping..."
fi

# Stage 2: lambeq Encoding
echo ""
echo "========================================="
echo "STAGE 2: LAMBEQ ENCODING"
echo "========================================="

if [ ! -f "/scratch/cbjp404/bradford_hackathon_2025/lambeq_embeddings/train.pt" ]; then
    echo "Running lambeq encoding..."
    python /scratch/cbjp404/bradford_hackathon_2025/lambeq_encoder.py
else
    echo "Embeddings already exist, skipping..."
fi

# Stage 3: Quixer Training
echo ""
echo "========================================="
echo "STAGE 3: MODEL TRAINING"
echo "========================================="

# Run Quixer with genomics dataset
python /scratch/cbjp404/bradford_hackathon_2025/run_genomics_training.py \
    --models Quixer LSTM Transformer \
    --device cuda \
    --embeddings_dir /scratch/cbjp404/bradford_hackathon_2025/lambeq_embeddings \
    --output_dir /scratch/cbjp404/bradford_hackathon_2025/results \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001

echo ""
echo "========================================="
echo "TRAINING COMPLETE"
echo "========================================="
echo "Results saved to: /scratch/cbjp404/bradford_hackathon_2025/results"
echo "Job finished at: $(date)"
