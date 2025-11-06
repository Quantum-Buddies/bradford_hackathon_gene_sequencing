# GPU Pipeline - Quick Start

## One Command to Run Everything

```bash
# Full GPU-accelerated pipeline (4-6 hours total)
bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 64 4 cuda && \
conda activate quixer && \
python quantize_lambeq_embeddings.py \
  --embeddings_dir lambeq_embeddings_autoregressive \
  --output_dir quantized_embeddings_autoregressive && \
python train_quixer_hybrid.py \
  --data_dir quantized_embeddings_autoregressive \
  --task autoregressive \
  --epochs 50 \
  --init_strategy small_gaussian \
  --use_layerwise_training
```

---

## Step-by-Step (Recommended)

### **Step 1: Prepare Data** (5 min, CPU)
```bash
python prepare_autoregressive_data.py \
  --fasta GRCh38_genomic_dataset/rna_sequences.fasta \
  --output_dir autoregressive_data
```

### **Step 2: GPU Encoding** (1-2 hours, GPU)
```bash
bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 64 4 cuda
```

**Monitor GPU**:
```bash
watch -n 1 nvidia-smi
```

### **Step 3: Quantize** (5 min, CPU)
```bash
conda activate quixer
python quantize_lambeq_embeddings.py \
  --embeddings_dir lambeq_embeddings_autoregressive \
  --output_dir quantized_embeddings_autoregressive
```

### **Step 4: Train** (2-3 hours, GPU)
```bash
python train_quixer_hybrid.py \
  --data_dir quantized_embeddings_autoregressive \
  --task autoregressive \
  --epochs 50 \
  --init_strategy small_gaussian \
  --use_layerwise_training
```

---

## GPU Options

```bash
# Default GPU (GPU 0)
bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 64 4 cuda

# Specific GPU
bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 64 4 cuda:1

# CPU (slow)
bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 64 1 cpu

# More workers (faster)
bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 64 8 cuda

# Larger embeddings (slower but better)
bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 128 4 cuda
```

---

## Expected Results

```
✅ Data preparation: ~5 min
✅ GPU encoding: ~1-2 hours (4-8× faster than CPU)
✅ Quantization: ~5 min
✅ Training: ~2-3 hours
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Total: ~4-6 hours
```

---

## Check GPU Status

```bash
# Check available GPUs
nvidia-smi

# Watch GPU during encoding
watch -n 1 nvidia-smi

# Check PyTorch GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce workers: `--workers 2` |
| GPU not detected | Check: `nvidia-smi` |
| Slow GPU | Check GPU-Util in `nvidia-smi` |
| Wrong GPU | Specify: `cuda:1` instead of `cuda` |

---

## Done! 🚀

After training completes, check results:
```bash
cat outputs/training_results.txt
```

Expected accuracy: **15-25%** (512-way next-token prediction)
Expected perplexity: **20-50** (lower is better)
