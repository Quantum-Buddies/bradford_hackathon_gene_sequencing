#!/bin/bash
#
# Run Full Hybrid Quantum Genomics Pipeline
# ==========================================
# Combines Lambeq + K-means + Quixer for double quantum advantage
#

set -e  # Exit on error

PYTHON=/scratch/cbjp404/conda_envs/quixer/bin/python
GPU=0

echo "========================================================================"
echo "HYBRID QUANTUM GENOMICS PIPELINE"
echo "========================================================================"
echo "Step 1: Lambeq quantum compositional embeddings"
echo "Step 2: Vector quantization (k-means clustering)"
echo "Step 3: Quixer quantum transformer"
echo "========================================================================"
echo ""

# Step 1: Quantize existing lambeq embeddings
echo "▶ Step 1: Quantizing lambeq embeddings..."
$PYTHON quantize_lambeq_embeddings.py \
  --embeddings_dir /scratch/cbjp404/bradford_hackathon_2025/lambeq_embeddings \
  --output_dir /scratch/cbjp404/bradford_hackathon_2025/quantized_embeddings \
  --n_clusters 512 \
  --seq_len 32 \
  --seed 42

echo ""
echo "✓ Quantization complete!"
echo ""

# Step 2: Train Quixer on quantized tokens
echo "▶ Step 2: Training Quixer on quantized quantum tokens..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON train_quixer_hybrid.py \
  --data_dir /scratch/cbjp404/bradford_hackathon_2025/quantized_embeddings \
  --output_dir /scratch/cbjp404/bradford_hackathon_2025/quixer_hybrid_results \
  --qubits 6 \
  --layers 3 \
  --ansatz_layers 4 \
  --embedding_dim 64 \
  --dropout 0.15 \
  --batch_size 32 \
  --epochs 50 \
  --lr 0.002 \
  --weight_decay 0.001 \
  --scheduler cosine \
  --device cuda

echo ""
echo "========================================================================"
echo "PIPELINE COMPLETE!"
echo "========================================================================"
echo "Results saved to: /scratch/cbjp404/bradford_hackathon_2025/quixer_hybrid_results/"
echo ""
echo "This hybrid approach combined:"
echo "  ✓ Lambeq quantum compositional embeddings"
echo "  ✓ K-means vector quantization"
echo "  ✓ Quixer quantum transformer"
echo ""
echo "= DOUBLE QUANTUM ADVANTAGE! ="
echo "========================================================================"
