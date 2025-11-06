# Pipeline Verification for Next-Token Prediction

## ✅ Downloaded Sequences Look Good!

Your RNA download worked perfectly:
```
Success: 10,000 sequences
Min: 235 bp
Max: 34,400 bp
Mean: 3,610 bp
```

This is **excellent** for next-token prediction:
- Mean of 3,610 bp = ~600 k-mers (with k=6)
- Each sequence will generate ~570 training windows (600-32+1)
- **Total expected windows: ~5.7 million** (10k sequences × 570 windows)

---

## Pipeline Flow & Compatibility

### **Step 1: prepare_autoregressive_data.py** ✅ FIXED

**Status**: Fixed to match lambeq expectations

**What it does**:
- Extracts 32 k-mer windows from your 10k RNA sequences
- Creates train/val/test splits with subdirectories
- Saves dummy labels (required by lambeq but not used)

**Fixed issues**:
- ✅ Now creates subdirectories: `autoregressive_data/train/sentences.txt`
- ✅ Creates dummy `labels.txt` files (all 0s)

**Run**:
```bash
python prepare_autoregressive_data.py \
  --fasta GRCh38_genomic_dataset/rna_sequences.fasta \
  --output_dir autoregressive_data \
  --k 6 \
  --window_size 32
```

**Expected output**:
```
autoregressive_data/
├── train/
│   ├── sentences.txt  (~4M windows)
│   └── labels.txt     (dummy, all 0s)
├── val/
│   ├── sentences.txt  (~850k windows)
│   └── labels.txt
├── test/
│   ├── sentences.txt  (~850k windows)
│   └── labels.txt
└── metadata.json
```

---

### **Step 2: lambeq_encoder.py** ✅ COMPATIBLE

**Status**: Works as-is, no changes needed

**Environment**: 
```bash
module load miniforge
conda activate qrisp-jax
```

**What it does**:
- Reads `train/sentences.txt`, `train/labels.txt`, etc.
- Encodes each k-mer with IQP circuits
- Saves per-k-mer embeddings: `[N, 32, 64]`

**Run**:
```bash
python lambeq_encoder.py \
  --data_dir autoregressive_data \
  --output_dir lambeq_embeddings_autoregressive \
  --embedding_dim 64 \
  --layers 2
```

**Output**:
```
lambeq_embeddings_autoregressive/
├── train.pt  (embeddings: [4M, 32, 64], labels: [4M])
├── val.pt    (embeddings: [850k, 32, 64], labels: [850k])
├── test.pt
└── metadata.json
```

**Note**: Labels will all be 0 (dummy), but that's fine - we don't use them!

---

### **Step 3: quantize_lambeq_embeddings.py** ✅ COMPATIBLE

**Status**: Works as-is, no changes needed

**What it does**:
- Clusters k-mer embeddings into 512 tokens
- Saves cluster centroids
- Converts embeddings to token sequences

**Run**:
```bash
python quantize_lambeq_embeddings.py \
  --embeddings_dir lambeq_embeddings_autoregressive \
  --output_dir quantized_embeddings_autoregressive \
  --n_clusters 512 \
  --seq_len 32
```

**Output**:
```
quantized_embeddings_autoregressive/
├── train.pt         (sequences: [4M, 32], labels: [4M])
├── val.pt           (sequences: [850k, 32], labels: [850k])
├── test.pt
├── cluster_centers.pt  (512 × 64)
└── metadata.json
```

**Critical**: `sequences` contains **full windows** (all 32 tokens)

---

### **Step 4: Training Scripts** ⚠️ NEEDS MODIFICATION

**Environment**:
```bash
conda activate quixer
```

#### **Current Setup (Binary Classification)**
```python
# In train_quixer_hybrid.py, train_epoch():
sequences, labels = batch  # [batch, 32], [batch]
outputs = model(sequences)  # [batch, 2] ← WRONG for next-token
loss = criterion(outputs, labels)  # labels are 0 or 1
```

#### **What We Need (Next-Token Prediction)**
```python
sequences, _ = batch  # [batch, 32], ignore labels
inputs = sequences[:, :-1]   # [batch, 31] - first 31 tokens
targets = sequences[:, -1]   # [batch] - 32nd token (what to predict)
outputs = model(inputs)      # [batch, 512] - logits over vocab
loss = criterion(outputs, targets)  # targets are 0-511
```

---

## Required Changes

### **Option A: Modify Training Loop (Recommended)** ✅

Add a flag `--task` to training scripts:

```python
# In train_quixer_hybrid.py, train_epoch()

def train_epoch(model, train_loader, optimizer, criterion, device, 
                task='classification', scheduler=None, epoch_num=None, total_epochs=None):
    ...
    for batch_idx, (sequences, labels) in enumerate(pbar):
        sequences = sequences.to(device)
        
        if task == 'autoregressive':
            # Next-token prediction
            inputs = sequences[:, :-1]   # [batch, 31]
            targets = sequences[:, -1]   # [batch]
        else:
            # Binary classification
            inputs = sequences
            targets = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        ...
```

Then in `main()`:
```python
parser.add_argument('--task', choices=['classification', 'autoregressive'], 
                    default='classification')

# In training loop
train_metrics = train_epoch(
    model, train_loader, optimizer, criterion, device,
    task=args.task,
    scheduler=scheduler, 
    epoch_num=epoch, 
    total_epochs=args.epochs
)
```

### **Option B: Modify quantize script to save properly**

Modify `quantize_lambeq_embeddings.py` to save data differently for autoregressive:

```python
# When saving autoregressive data:
torch.save({
    'sequences': token_sequences,  # [N, 32]
    'inputs': token_sequences[:, :-1],   # [N, 31] 
    'targets': token_sequences[:, -1],   # [N]
    'task': 'autoregressive'
}, output_path)
```

Then training scripts auto-detect the task.

---

## Files Status

| File | Status | Changes Needed |
|------|--------|----------------|
| `download_rna_sequences.py` | ✅ Working | None (already ran) |
| `prepare_autoregressive_data.py` | ✅ Fixed | None (already fixed) |
| `lambeq_encoder.py` | ✅ Compatible | None |
| `quantize_lambeq_embeddings.py` | ✅ Compatible | None |
| `train_quixer_hybrid.py` | ⚠️ Needs update | Add `--task` flag |
| `tune_quixer_hybrid.py` | ⚠️ Needs update | Add `--task` flag |
| `quixer_wrapper.py` | ✅ Correct | None (output=512 is correct!) |
| `hybrid_quixer_classifier.py` | ❌ Not needed | Delete or ignore |

---

## Do You Need hybrid_quixer_classifier.py?

**NO** - You can delete or ignore it.

**Why it exists**: Pure PyTorch fallback for when TorchQuantum isn't available

**What you're using**: `quixer_wrapper.py` → Real TorchQuantum Quixer ✅

---

## Next Steps

### **Immediate (30 minutes)**

1. **Prepare data**:
```bash
python prepare_autoregressive_data.py \
  --fasta GRCh38_genomic_dataset/rna_sequences.fasta \
  --output_dir autoregressive_data
```

2. **Encode with lambeq** (may take 1-2 hours):
```bash
module load miniforge
conda activate qrisp-jax

python lambeq_encoder.py \
  --data_dir autoregressive_data \
  --output_dir lambeq_embeddings_autoregressive \
  --embedding_dim 64
```

3. **Quantize** (fast, ~5 minutes):
```bash
conda activate quixer

python quantize_lambeq_embeddings.py \
  --embeddings_dir lambeq_embeddings_autoregressive \
  --output_dir quantized_embeddings_autoregressive \
  --n_clusters 512 \
  --seq_len 32
```

4. **Modify training scripts** (I'll do this for you in next response)

5. **Train**:
```bash
python tune_quixer_hybrid.py \
  --data_dir quantized_embeddings_autoregressive \
  --task autoregressive \
  --n_trials 30 \
  --epochs_per_trial 50
```

---

## Expected Results

### **Metrics**

For next-token prediction with 512-way classification:

| Metric | Random Baseline | Expected | Excellent |
|--------|----------------|----------|-----------|
| **Accuracy** | 0.2% (1/512) | 15-25% | 30%+ |
| **Top-5 Accuracy** | 1% | 40-50% | 60%+ |
| **Perplexity** | 512 | 20-50 | <20 |

**Why lower accuracy is OK?**
- 512-way classification is MUCH harder than 2-way (binary)
- Multiple k-mers might be valid next tokens
- Perplexity is the better metric

### **Comparison to Original Quixer**

Original paper (Penn Treebank, 10k vocab):
- Perplexity: ~114
- Competitive with LSTM

Your setup (RNA sequences, 512 vocab):
- Expected perplexity: 20-50 (easier vocab → better perplexity)
- With optimizations: **Should beat LSTM!**

---

## Summary

✅ **Data pipeline**: All compatible, minor fixes applied
✅ **Architecture**: Perfect as-is (output=512 is correct!)
⚠️ **Training loop**: Need to add `--task autoregressive` flag

**Bottom line**: You're 95% there! Just need small training loop modification.

Let me know if you want me to implement the training script changes!
