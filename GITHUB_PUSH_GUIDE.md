# GitHub Push Guide - Next-Token Prediction Pipeline

## 📋 What's Ready to Push

### **New Files Created**
1. ✅ `prepare_autoregressive_data.py` - Autoregressive data preparation
2. ✅ `prepare_classical_benchmarks.py` - Classical baseline data prep
3. ✅ `run_lambeq_gpu.sh` - GPU-accelerated lambeq launcher
4. ✅ `run_classical_prep.sh` - Classical data preparation launcher
5. ✅ `LAMBEQ_GPU_GUIDE.md` - GPU acceleration documentation
6. ✅ `GPU_QUICK_START.md` - Quick reference
7. ✅ `PIPELINE_VERIFICATION.md` - Pipeline verification guide
8. ✅ `CHANGES_SUMMARY.md` - Summary of changes
9. ✅ `NEXT_TOKEN_PREDICTION_GUIDE.md` - Complete guide

### **Modified Files**
1. ✅ `train_quixer_hybrid.py` - Added `--task` argument for autoregressive
2. ✅ `lambeq_encoder.py` - Added `--data_dir` and `--output_dir` arguments

### **Data Files** (Optional - Large)
- `GRCh38_genomic_dataset/rna_sequences.fasta` - 10k RNA sequences (~50MB)
- `autoregressive_data/` - Prepared autoregressive windows
- `classical_benchmarks_data/` - Classical baseline data
- `lambeq_embeddings_autoregressive/` - Lambeq-encoded embeddings
- `quantized_embeddings_autoregressive/` - Quantized token sequences

---

## 🚀 GitHub Push Steps

### **Step 1: Check Git Status**

```bash
cd /scratch/cbjp404/bradford_hackathon_gene_sequencing
git status
```

### **Step 2: Add Files**

```bash
# Add all new Python scripts
git add prepare_autoregressive_data.py
git add prepare_classical_benchmarks.py
git add lambeq_encoder.py  # Modified
git add train_quixer_hybrid.py  # Modified

# Add all shell scripts
git add run_lambeq_gpu.sh
git add run_classical_prep.sh

# Add all documentation
git add LAMBEQ_GPU_GUIDE.md
git add GPU_QUICK_START.md
git add PIPELINE_VERIFICATION.md
git add CHANGES_SUMMARY.md
git add NEXT_TOKEN_PREDICTION_GUIDE.md
git add GITHUB_PUSH_GUIDE.md

# Optional: Add data files (if not too large)
# git add GRCh38_genomic_dataset/rna_sequences.fasta
# git add classical_benchmarks_data/
```

### **Step 3: Commit**

```bash
git commit -m "Add autoregressive next-token prediction pipeline with GPU support

- Implement autoregressive data preparation for RNA sequences
- Add GPU-accelerated lambeq encoding (4-8x speedup)
- Create classical benchmark data preparation (LSTM, Transformer)
- Add --task argument to train_quixer_hybrid.py for autoregressive training
- Fix lambeq_encoder.py to accept --data_dir and --output_dir arguments
- Comprehensive documentation and quick-start guides
- Expected accuracy: 15-25% (512-way next-token prediction)
- Expected perplexity: 20-50 (vs 512 random baseline)

Files:
- prepare_autoregressive_data.py: Converts RNA sequences to k-mer windows
- prepare_classical_benchmarks.py: Prepares data for LSTM/Transformer baselines
- run_lambeq_gpu.sh: GPU-accelerated lambeq encoding launcher
- run_classical_prep.sh: Classical benchmark data preparation launcher
- Modified lambeq_encoder.py: Added command-line arguments
- Modified train_quixer_hybrid.py: Added autoregressive task support
- Documentation: 5 comprehensive guides for GPU pipeline"
