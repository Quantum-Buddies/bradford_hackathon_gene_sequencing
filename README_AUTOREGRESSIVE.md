# Quixer Next-Token Prediction Pipeline

## Overview

This pipeline implements **autoregressive next-token prediction** using Quixer quantum transformer on real human RNA sequences from GRCh38.

**Task**: Given k-mers 1 to 31, predict k-mer 32 (512-way classification)

**Hardware**: 1× NVIDIA L40S (47.7GB) or 2× L40 (48GB each)

**Expected Results**:
- Accuracy: 15-25% (vs 0.2% random baseline)
- Perplexity: 20-50 (vs 512 random)
- Training time: 4-6 hours with GPU

---

## Quick Start (4-6 Hours)

### **1. Download RNA Sequences** (Already Done ✅)
```bash
python download_rna_sequences.py --max_sequences 10000
# Output: 10,000 real RNA sequences from NCBI
```

### **2. Prepare Autoregressive Data** (5 min)
```bash
python prepare_autoregressive_data.py \
  --fasta GRCh38_genomic_dataset/rna_sequences.fasta \
  --output_dir autoregressive_data
```

### **3. GPU-Accelerated Lambeq Encoding** (1-2 hours)
```bash
bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 64 8 cuda
```

### **4. Quantize Embeddings** (5 min)
```bash
conda activate quixer
python quantize_lambeq_embeddings.py \
  --embeddings_dir lambeq_embeddings_autoregressive \
  --output_dir quantized_embeddings_autoregressive
```

### **5. Train Quixer** (2-3 hours)
```bash
python train_quixer_hybrid.py \
  --data_dir quantized_embeddings_autoregressive \
  --task autoregressive \
  --epochs 50 \
  --init_strategy small_gaussian \
  --use_layerwise_training
```

---

## Pipeline Architecture

```
RNA Sequences (10k)
    ↓
prepare_autoregressive_data.py
    ↓
K-mer windows (32 tokens each)
    ├─→ lambeq_encoder.py (GPU) ──→ Per-k-mer embeddings [N, 32, 64]
    │
    ├─→ prepare_classical_benchmarks.py ──→ Token sequences [N, 32]
    │
    ↓
quantize_lambeq_embeddings.py
    ↓
Token IDs [N, 32] + Cluster centroids
    ├─→ train_quixer_hybrid.py (Quixer)
    │   Input: [N, 31] → Output: [N, 512] logits
    │   Target: [N] (token 32)
    │
    └─→ train_classical_baselines.py (LSTM, Transformer)
        Input: [N, 31] → Output: [N, 512] logits
        Target: [N] (token 32)
```

---

## Data Format

### **Input: K-mer Sentences**
```
autoregressive_data/
├── train/
│   ├── sentences.txt  (k-mer sentences, space-separated)
│   └── labels.txt     (dummy labels, all 0s)
├── val/
└── test/
```

Example sentence:
```
ACGTAC CGTACG GTACGG TACGGT ACGGTC ... (32 k-mers)
```

### **After Lambeq Encoding**
```
lambeq_embeddings_autoregressive/
├── train.pt  (embeddings: [N, 32, 64], labels: [N])
├── val.pt
└── test.pt
```

### **After Quantization**
```
quantized_embeddings_autoregressive/
├── train.pt  (sequences: [N, 32], labels: [N])
├── val.pt
├── test.pt
└── cluster_centers.pt  (512 × 64)
```

---

## Training Details

### **Task: Next-Token Prediction**

```python
# Input: First 31 k-mers
inputs = sequences[:, :-1]   # [batch, 31]

# Target: 32nd k-mer (token ID 0-511)
targets = sequences[:, -1]   # [batch]

# Forward pass
outputs = model(inputs)      # [batch, 512] logits

# Loss
loss = criterion(outputs, targets)  # CrossEntropyLoss
```

### **Key Hyperparameters**

| Parameter | Value | Reason |
|-----------|-------|--------|
| `--task` | `autoregressive` | Next-token prediction |
| `--init_strategy` | `small_gaussian` | Avoid barren plateaus |
| `--use_layerwise_training` | True | Gradual parameter training |
| `--epochs` | 50 | Quantum circuits need time |
| `--batch_size` | 32 | Balance memory/gradients |
| `--lr` | 0.002 | Conservative learning rate |

---

## Classical Baselines

### **Prepare Benchmark Data**
```bash
bash run_classical_prep.sh GRCh38_genomic_dataset/rna_sequences.fasta classical_benchmarks_data
```

### **Train LSTM Baseline** (Coming Soon)
```bash
python train_classical_baselines.py \
  --data_dir classical_benchmarks_data \
  --model lstm \
  --epochs 100
```

### **Train Transformer Baseline** (Coming Soon)
```bash
python train_classical_baselines.py \
  --data_dir classical_benchmarks_data \
  --model transformer \
  --epochs 100
```

---

## GPU Acceleration

### **Lambeq on GPU**

Lambeq encoding is **4-8× faster on GPU**:

```bash
# GPU (1-2 hours)
bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 64 8 cuda

# CPU (8-12 hours)
bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 64 1 cpu
```

### **Monitor GPU**
```bash
watch -n 1 nvidia-smi
```

Expected:
- GPU-Util: 70-90%
- Memory: 20-30GB
- Temp: <80°C

---

## Expected Results

### **Quixer (Quantum)**
- Accuracy: 15-25%
- Perplexity: 20-50
- Parameters: ~50k
- Training time: 2-3 hours

### **LSTM Baseline**
- Accuracy: 20-30%
- Perplexity: 15-30
- Parameters: ~200k
- Training time: 1-2 hours

### **Transformer Baseline**
- Accuracy: 25-35%
- Perplexity: 12-25
- Parameters: ~500k
- Training time: 2-4 hours

### **Random Baseline**
- Accuracy: 0.2% (1/512)
- Perplexity: 512

---

## Troubleshooting

### **Issue: "CUDA out of memory"**
```bash
# Reduce workers
bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 64 4 cuda

# Or reduce embedding dim
bash run_lambeq_gpu.sh autoregressive_data lambeq_embeddings_autoregressive 32 8 cuda
```

### **Issue: "Accuracy only 0.2%"**
Check if `--task autoregressive` is set in training command.

### **Issue: "Slow GPU performance"**
Check `nvidia-smi` - GPU-Util should be 70-90%.

---

## Files Overview

### **Data Preparation**
- `prepare_autoregressive_data.py` - Autoregressive window extraction
- `prepare_classical_benchmarks.py` - Classical baseline data prep
- `download_rna_sequences.py` - Download RNA from NCBI

### **Encoding & Quantization**
- `lambeq_encoder.py` - Quantum circuit encoding (GPU-accelerated)
- `quantize_lambeq_embeddings.py` - K-means clustering

### **Training**
- `train_quixer_hybrid.py` - Quixer training (autoregressive support)
- `tune_quixer_hybrid.py` - Hyperparameter tuning
- `train_classical_baselines.py` - LSTM/Transformer training (coming soon)

### **Launchers**
- `run_lambeq_gpu.sh` - GPU encoding launcher
- `run_classical_prep.sh` - Classical data prep launcher

### **Documentation**
- `README_AUTOREGRESSIVE.md` - This file
- `LAMBEQ_GPU_GUIDE.md` - GPU acceleration guide
- `GPU_QUICK_START.md` - Quick reference
- `NEXT_TOKEN_PREDICTION_GUIDE.md` - Detailed guide

---

## References

1. **Quixer Paper**: arXiv:2406.04305 (LCU+QSVT quantum attention)
2. **lambeq Documentation**: https://docs.quantinuum.com/lambeq/
3. **Next-Token Prediction**: Cameron Wolfe's LLM guide
4. **GRCh38 RNA Data**: NCBI RefSeq (184k+ mRNA transcripts)

---

## Citation

If you use this pipeline, please cite:

```bibtex
@article{quixer2024,
  title={Quixer: A Quantum Transformer Model},
  author={...},
  journal={arXiv},
  year={2024}
}
```

---

## License

This project is part of the Bradford Hackathon 2025.

---

## Contact

For questions or issues, please open a GitHub issue.

Good luck! 🚀
