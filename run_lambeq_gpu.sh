#!/bin/bash
#
# GPU-Accelerated Lambeq Encoding Pipeline
# ==========================================
# Encodes k-mer sequences using quantum circuits on GPU
#
# Usage:
#   bash run_lambeq_gpu.sh [data_dir] [output_dir] [embedding_dim] [workers] [device]
#
# Examples:
#   bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 64 4 cuda:0
#   bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 64 8 cuda
#

set -e

# Default arguments
DATA_DIR="${1:-autoregressive_data}"
OUTPUT_DIR="${2:-lambeq_embeddings_autoregressive}"
EMBEDDING_DIM="${3:-64}"
WORKERS="${4:-4}"
DEVICE="${5:-cuda}"

echo "=========================================="
echo "GPU-ACCELERATED LAMBEQ ENCODING"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Embedding dimension: $EMBEDDING_DIM"
echo "  Workers: $WORKERS"
echo "  Device: $DEVICE"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo "Run prepare_autoregressive_data.py first"
    exit 1
fi

# Activate environment
echo "Activating qrisp-jax environment..."
module load miniforge
conda activate qrisp-jax

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB')
else:
    print('WARNING: CUDA not available, will use CPU')
"

echo ""
echo "Starting lambeq encoding on $DEVICE..."
echo ""

# Run lambeq encoder with GPU support
python lambeq_encoder.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --embedding_dim "$EMBEDDING_DIM" \
    --layers 2 \
    --workers "$WORKERS" \
    --parser_device "$DEVICE"

echo ""
echo "=========================================="
echo "✅ Lambeq encoding complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Quantize embeddings:"
echo "   python quantize_lambeq_embeddings.py \\"
echo "     --embeddings_dir $OUTPUT_DIR \\"
echo "     --output_dir quantized_embeddings_autoregressive"
echo ""
echo "2. Train Quixer:"
echo "   conda activate quixer"
echo "   python train_quixer_hybrid.py \\"
echo "     --data_dir quantized_embeddings_autoregressive \\"
echo "     --task autoregressive"
echo ""
