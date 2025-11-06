# Next-Token Prediction with Quixer

## Your Current Dataset

You have **metadata CSVs** from GRCh38:
- `GRCh38_latest_rna_summary.csv` – 184,491 mRNA transcript IDs
- `GRCh38_latest_genomic_summary.csv` – Chromosome metadata
- `GRCh38_latest_protein_symmery.csv` – Protein metadata

**What's missing**: Actual DNA/RNA sequences (FASTA files)

---

## 3 Options for Next-Token Prediction

### **Option 1: Download Real RNA Sequences** ✅ RECOMMENDED

Use the 184k+ mRNA IDs you already have to fetch real sequences from NCBI.

#### **Step 1: Download Sequences**

```bash
# Install biopython if needed
conda install -c conda-forge biopython

# Edit download_rna_sequences.py and set your email (line 13)
nano download_rna_sequences.py
# Change: Entrez.email = "your.email@example.com"

# Download sequences (start with 1000 to test)
python download_rna_sequences.py \
  --csv GRCh38_genomic_dataset/GRCh38_latest_rna_summary.csv \
  --output GRCh38_genomic_dataset/rna_sequences.fasta \
  --max_sequences 1000 \
  --batch_size 50
```

**Time**: ~10 minutes for 1000 sequences, ~3 hours for 10k

#### **Step 2: Prepare Autoregressive Windows**

```bash
python prepare_autoregressive_data.py \
  --fasta GRCh38_genomic_dataset/rna_sequences.fasta \
  --output_dir autoregressive_data \
  --k 6 \
  --window_size 32 \
  --min_length 200
```

**Output**:
- `autoregressive_data/train_sentences.txt` – K-mer sentences
- `autoregressive_data/val_sentences.txt`
- `autoregressive_data/test_sentences.txt`
- `autoregressive_data/metadata.json`

#### **Step 3: Encode with Lambeq**

```bash
python lambeq_encoder.py \
  --data_dir autoregressive_data \
  --output_dir lambeq_embeddings_autoregressive \
  --k 6 \
  --max_kmers 32 \
  --embedding_dim 64
```

#### **Step 4: Quantize Embeddings**

```bash
python quantize_lambeq_embeddings.py \
  --embeddings_dir lambeq_embeddings_autoregressive \
  --output_dir quantized_embeddings_autoregressive \
  --n_clusters 512 \
  --seq_len 32
```

#### **Step 5: Train Quixer (No Architecture Change Needed!)**

Your current model is **already correct** for next-token prediction:

```bash
# Tune hyperparameters
python tune_quixer_hybrid.py \
  --data_dir quantized_embeddings_autoregressive \
  --n_trials 30 \
  --epochs_per_trial 50 \
  --init_strategy small_gaussian \
  --use_layerwise_training

# Train final model
python train_quixer_hybrid.py \
  --data_dir quantized_embeddings_autoregressive \
  --epochs 100 \
  --init_strategy small_gaussian \
  --use_layerwise_training
```

**Why no changes?**
- Output: 512 logits (vocabulary size) ✅
- Target: Token ID 0-511 ✅
- Loss: CrossEntropyLoss ✅

The model predicts which k-mer comes next!

---

### **Option 2: Use Synthetic Sequences (Quick Test)**

Your current `preprocess_genomics.py` generates synthetic data. Adapt it for next-token prediction:

```python
# In preprocess_genomics.py, modify create_synthetic_labels():

def create_autoregressive_windows(
    self,
    n_sequences: int = 10000
) -> List[str]:
    """Generate random sequences for next-token prediction."""
    sequences = []
    for _ in range(n_sequences):
        # Generate random 512bp sequence
        seq = ''.join(np.random.choice(['A', 'C', 'G', 'T'], size=512))
        sequences.append(seq)
    return sequences
```

Then run the same pipeline (encode → quantize → train).

---

### **Option 3: Download Full Genome (Advanced)**

Download entire human genome from UCSC:

```bash
# Download chromosome 1 only (249 MB compressed)
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr1.fa.gz
gunzip chr1.fa.gz

# Or all chromosomes (938 MB compressed)
wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz
```

Then use `prepare_autoregressive_data.py` with the FASTA file.

---

## Task: Next-Token Prediction

### **Input Format**
```
Sequence: ACGTAC CGTACG GTACGG TACGGT ACGGTC ...
          [k1]   [k2]   [k3]   [k4]   [k5]

Window: [k1, k2, k3, ..., k31] → Predict k32
```

### **Data Format After Quantization**
```python
# quantized_embeddings_autoregressive/train.pt
{
    'sequences': tensor([[tok1, tok2, ..., tok32],  # [N, 32]
                        [tok1, tok2, ..., tok32],
                        ...]),
    'labels': None  # Not used for next-token prediction
}
```

### **Training Loop**
```python
# Extract input and target from same sequence
sequences = batch['sequences']  # [batch, 32]
inputs = sequences[:, :-1]      # [batch, 31] - first 31 tokens
targets = sequences[:, -1]      # [batch] - 32nd token

# Forward pass
outputs = model(inputs)         # [batch, 512] logits

# Loss
loss = criterion(outputs, targets)  # CrossEntropyLoss

# Metrics
_, predicted = outputs.max(1)
accuracy = (predicted == targets).float().mean()
perplexity = torch.exp(loss)
```

---

## Evaluation Metrics

For next-token prediction, use:

1. **Accuracy**: % of correctly predicted tokens
2. **Perplexity**: `exp(cross_entropy_loss)` (lower = better)
   - Good: 10-20
   - Excellent: <10
3. **Top-k Accuracy**: % where true token is in top-k predictions
   - Top-5 accuracy often reported

---

## Expected Results

### **Current Setup (Binary Classification)**
- Task: Promoter vs non-promoter
- Accuracy: ~56% (near-random)
- Issue: Output dimension mismatch (512 vs 2 classes)

### **Next-Token Prediction (Correct Setup)**
- Task: Predict next k-mer from 512 vocab
- Expected accuracy: 
  - Random baseline: 0.2% (1/512)
  - Simple model: 5-10%
  - Optimized Quixer: **15-25%** (with all optimizations)
- Expected perplexity: 20-50

**Why lower accuracy is OK?**
- 512-way classification is HARD (vs 2-way)
- Multiple valid next tokens possible
- Perplexity is better metric than accuracy

---

## Comparison to Quixer Paper

Original Quixer paper (arXiv:2406.04305):
- **Dataset**: Penn Treebank (PTB)
- **Task**: Next word prediction
- **Vocabulary**: ~10k words
- **Results**: Perplexity ~114 (competitive with LSTM)

Your setup:
- **Dataset**: Human RNA sequences
- **Task**: Next k-mer prediction
- **Vocabulary**: 512 k-mers
- **Expected**: Perplexity 20-50 (easier vocab)

---

## Quick Start (30 Minutes)

```bash
# 1. Download 1000 RNA sequences
python download_rna_sequences.py --max_sequences 1000

# 2. Prepare autoregressive windows
python prepare_autoregressive_data.py \
  --fasta GRCh38_genomic_dataset/rna_sequences.fasta \
  --output_dir autoregressive_data

# 3. Encode with lambeq
python lambeq_encoder.py \
  --data_dir autoregressive_data \
  --output_dir lambeq_embeddings_autoregressive

# 4. Quantize
python quantize_lambeq_embeddings.py \
  --embeddings_dir lambeq_embeddings_autoregressive \
  --output_dir quantized_embeddings_autoregressive

# 5. Train Quixer
python tune_quixer_hybrid.py \
  --data_dir quantized_embeddings_autoregressive \
  --n_trials 10 \
  --epochs_per_trial 20 \
  --init_strategy small_gaussian
```

---

## Summary

| Aspect | Binary Classification | Next-Token Prediction |
|--------|----------------------|----------------------|
| **Task** | Promoter vs non-promoter | Predict next k-mer |
| **Output** | 2 classes | 512 classes |
| **Data** | Synthetic (motif injection) | Real RNA sequences |
| **Accuracy** | ~56% (broken) | 15-25% (512-way) |
| **Better metric** | F1-score | Perplexity |
| **Architecture** | Need classifier head | Current is correct ✅ |
| **Dataset source** | `preprocess_genomics.py` | NCBI RNA sequences |

**Bottom line**: For next-token prediction, your architecture is perfect as-is. Just need real sequences!

---

## Troubleshooting

**Q: Download is too slow?**
- Start with 1000 sequences to test
- Use `--batch_size 50` for faster downloads
- Or use synthetic data for quick prototyping

**Q: Do I need to modify training scripts?**
- No! Current scripts work for both tasks
- Only difference: data format (classification vs autoregressive)

**Q: What about the "bug" you mentioned?**
- That only applies to binary classification (2 classes)
- For next-token (512 classes), output dimension is correct

Good luck! 🚀
