#!/bin/bash
#
# Prepare Classical Benchmark Data
# ================================
# Creates datasets for LSTM, Transformer baselines
#
# Usage:
#   bash run_classical_prep.sh [fasta_file] [output_dir]
#

set -e

FASTA="${1:-GRCh38_genomic_dataset/rna_sequences.fasta}"
OUTPUT_DIR="${2:-classical_benchmarks_data}"

echo "=========================================="
echo "CLASSICAL BENCHMARK DATA PREPARATION"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  FASTA file: $FASTA"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Check if FASTA exists
if [ ! -f "$FASTA" ]; then
    echo "ERROR: FASTA file not found: $FASTA"
    exit 1
fi

# Activate environment with Bio
echo "Activating qrisp-jax environment..."
module load miniforge
conda activate qrisp-jax

echo ""
echo "Starting classical benchmark data preparation..."
echo ""

# Run preparation
python prepare_classical_benchmarks.py \
    --fasta "$FASTA" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "✅ Classical benchmark data ready!"
echo "=========================================="
echo ""
