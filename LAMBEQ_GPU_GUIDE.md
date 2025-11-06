# Lambeq GPU Acceleration Guide

## ✅ GPU Support Already Built In

Your `lambeq_encoder.py` already has full GPU support via the `--parser_device` argument!

```python
# Line 195 in lambeq_encoder.py:
self.parser = BobcatParser(verbose='text', device=self.device)
```

---

## Quick Start: GPU Encoding

### **Option 1: Automatic (Recommended)**

```bash
bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 64 4 cuda
```

**What this does**:
- ✅ Loads qrisp-jax environment
- ✅ Checks GPU availability
- ✅ Runs lambeq with GPU acceleration
- ✅ Shows next steps

### **Option 2: Manual Command**

```bash
# Activate environment
module load miniforge
conda activate qrisp-jax

# Run with GPU
python lambeq_encoder.py \
  --data_dir autoregressive_data \
  --output_dir lambeq_embeddings_autoregressive \
  --embedding_dim 64 \
  --layers 2 \
  --workers 4 \
  --parser_device cuda
```

---

## GPU Device Options

### **Single GPU (Default)**
```bash
--parser_device cuda          # Uses GPU 0 (default)
--parser_device cuda:0        # Explicitly GPU 0
```

### **Specific GPU**
```bash
--parser_device cuda:1        # Use GPU 1
--parser_device cuda:2        # Use GPU 2
```

### **CPU Fallback**
```bash
--parser_device cpu           # Use CPU (slower)
```

---

## Performance Tuning

### **Worker Threads**

More workers = faster processing (up to CPU core count):

```bash
# Slow (1 worker)
--workers 1

# Fast (4 workers)
--workers 4

# Very fast (8 workers, if you have 8+ cores)
--workers 8
```

**Recommendation**: Use `--workers = min(8, num_cpu_cores)`

### **Embedding Dimension**

Larger embeddings = more expressive but slower:

```bash
--embedding_dim 32    # Fast, less expressive
--embedding_dim 64    # Balanced (recommended)
--embedding_dim 128   # Slow, more expressive
```

### **Circuit Layers**

More layers = more quantum depth but slower:

```bash
--layers 1            # Fast, shallow circuits
--layers 2            # Balanced (recommended)
--layers 3            # Slow, deep circuits
```

---

## Expected Performance

### **Hardware: 2× L40 GPUs (48GB each)**

| Config | Time | Memory |
|--------|------|--------|
| `--embedding_dim 64 --layers 2 --workers 4` | ~1-2 hours | ~20GB |
| `--embedding_dim 128 --layers 3 --workers 8` | ~3-4 hours | ~35GB |
| `--embedding_dim 32 --layers 1 --workers 4` | ~30 min | ~10GB |

**For 10k RNA sequences with 32 k-mers each = ~320k windows**

### **Speedup vs CPU**

- **GPU (cuda)**: ~1-2 hours
- **CPU**: ~8-12 hours
- **Speedup**: 4-8× faster on GPU

---

## Complete Pipeline with GPU

### **Step 1: Prepare Data** (CPU, ~5 min)
```bash
python prepare_autoregressive_data.py \
  --fasta GRCh38_genomic_dataset/rna_sequences.fasta \
  --output_dir autoregressive_data
```

### **Step 2: Encode with GPU** (GPU, ~1-2 hours)
```bash
bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 64 4 cuda
```

### **Step 3: Quantize** (CPU, ~5 min)
```bash
conda activate quixer
python quantize_lambeq_embeddings.py \
  --embeddings_dir lambeq_embeddings_autoregressive \
  --output_dir quantized_embeddings_autoregressive
```

### **Step 4: Train Quixer** (GPU, ~2-3 hours)
```bash
python train_quixer_hybrid.py \
  --data_dir quantized_embeddings_autoregressive \
  --task autoregressive \
  --epochs 50 \
  --init_strategy small_gaussian \
  --use_layerwise_training
```

**Total time: ~4-6 hours with GPU** ✅

---

## Monitoring GPU Usage

### **During Encoding**

In another terminal:
```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or one-time check
nvidia-smi
```

**Expected output**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.00                 Driver Version: 550.00                   |
|-------------------------------+----------------------+----------------------+
| GPU  Name                Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA L40                   On   | 00:1E.0     Off |                    0 |
|  0%   30C    P2    45W / 300W   |   8500MiB / 46080MiB |     85%      Default |
+-------------------------------+----------------------+----------------------+
```

### **Key Metrics**

- **GPU-Util**: Should be 70-90% (good GPU usage)
- **Memory-Usage**: Should be 50-80% of total (not maxed out)
- **Temp**: Should stay <80°C (normal operation)

---

## Troubleshooting

### **Issue: "CUDA out of memory"**

**Solutions**:
1. Reduce workers: `--workers 2` (instead of 4)
2. Reduce embedding dim: `--embedding_dim 32` (instead of 64)
3. Reduce layers: `--layers 1` (instead of 2)
4. Use CPU: `--parser_device cpu`

### **Issue: "CUDA not available"**

**Check**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**If False**:
1. Check GPU drivers: `nvidia-smi`
2. Check PyTorch installation: `python -c "import torch; print(torch.__version__)"`
3. Reinstall PyTorch with CUDA support

### **Issue: "Slow GPU performance"**

**Check**:
1. Is GPU actually being used? Run `nvidia-smi` while encoding
2. Are workers set too high? Try `--workers 4`
3. Is GPU shared? Check `nvidia-smi` for other processes

---

## Advanced: Multi-GPU Encoding

### **Using Both L40 GPUs**

Lambeq's BobcatParser doesn't support multi-GPU directly, but you can:

**Option 1: Split data across GPUs**
```bash
# Terminal 1: Encode first half on GPU 0
python lambeq_encoder.py \
  --data_dir autoregressive_data \
  --output_dir lambeq_embeddings_part1 \
  --parser_device cuda:0 &

# Terminal 2: Encode second half on GPU 1
python lambeq_encoder.py \
  --data_dir autoregressive_data \
  --output_dir lambeq_embeddings_part2 \
  --parser_device cuda:1 &

# Then merge outputs
```

**Option 2: Use single GPU (simpler)**
```bash
# Just use one GPU, it's fast enough
bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 64 4 cuda:0
```

---

## Summary

| Aspect | CPU | GPU |
|--------|-----|-----|
| **Time (10k sequences)** | 8-12 hours | 1-2 hours |
| **Speedup** | Baseline | 4-8× |
| **Memory** | ~5GB | ~20GB |
| **Cost** | Free | Electricity |
| **Recommended** | No | ✅ Yes |

**Bottom line**: GPU encoding is 4-8× faster and uses less wall-clock time. Highly recommended!

---

## Quick Commands

```bash
# GPU encoding (recommended)
bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 64 4 cuda

# CPU encoding (slow)
bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 64 1 cpu

# GPU 1 specifically
bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 64 4 cuda:1

# Custom settings
python lambeq_encoder.py \
  --data_dir autoregressive_data \
  --output_dir lambeq_embeddings_autoregressive \
  --embedding_dim 128 \
  --layers 3 \
  --workers 8 \
  --parser_device cuda
```

Good luck! 🚀
