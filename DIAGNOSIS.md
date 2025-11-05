# Quixer Genomics Overfitting: Root Cause Analysis

## Problem Summary
Quixer is severely overfitting on the genomics classification task:
- Train accuracy: 50% → 93% by epoch 6
- Val accuracy: stuck at ~50% (random chance)
- Val loss: exploding from 0.69 → 1.31

## Root Cause: Signal Not in Truncated Sequences

### Key Finding from Diagnostic Analysis

Running `diagnose_data.py` revealed the critical issue:

```
Positive samples (should contain motifs): 3499
  First 32 k-mers: 390/3499 (11.1%)     ← ONLY 11% HAVE SIGNAL!
  First 64 k-mers: 729/3499 (20.8%)
  First 128 k-mers: 1357/3499 (38.8%)
  All k-mers: 3499/3499 (100.0%)

Negative samples (should NOT contain motifs): 3501
  First 32 k-mers: 181/3501 (5.2%)
  First 64 k-mers: 356/3501 (10.2%)
  All k-mers: 1984/3501 (56.7%)        ← 57% FALSE POSITIVES!
```

### What This Means

1. **The synthetic data has 507 k-mers per sequence** (from 512bp windows with k=6)
2. **Quixer training uses only the first 32 k-mers** (`--max_seq_len 32`)
3. **Only 11% of promoter sequences have the motif in those first 32 k-mers**
4. **The labels appear random to the model** → it memorizes noise instead of learning

### Why It Happens

The preprocessing (`preprocess_genomics.py`) creates synthetic labels by:
1. Generating random 512bp sequences
2. Injecting a 6bp promoter motif at a **random position** for positives
3. The motif can be anywhere in the 512bp window
4. When we truncate to first 32 k-mers (192bp), we lose 62% of the sequence
5. Result: **89% of positive samples look identical to negatives**

## Baseline Classifier Results

Logistic regression on **full k-mer counts** (all 507 k-mers):
- Train: **98.3%** accuracy (learns the signal!)
- Val: **61.0%** accuracy
- Test: **61.5%** accuracy

**Top features** correctly identify the injected motifs:
```
TTGACA: 5.56    ← promoter motif
TATAAA: 5.44    ← promoter motif
GGGCGG: 5.19    ← promoter motif
```

This proves the task **IS learnable** when using the full sequence, but:
- Still shows overfitting (98% train vs 61% val)
- 57% of "negative" samples contain motifs by random chance
- The synthetic labels are inherently noisy

## Why Quixer Overfits Worse Than Logistic Regression

1. **Higher capacity**: 267K-536K parameters vs 4100 parameters
2. **Sees even less signal**: Only first 32 k-mers (11% signal) vs all k-mers (100% signal)
3. **Wrong inductive bias**: Quantum transformers expect compositional structure (like language), but random DNA doesn't have this
4. **Embedding overhead**: Learning embeddings for 4100 sparse tokens wastes capacity

## Solutions (In Order of Impact)

### 1. **IMMEDIATE FIX: Increase Sequence Length**

```bash
--max_seq_len 128    # Captures 38.8% of positive signals (vs 11%)
--max_seq_len 256    # Captures ~70% of positive signals
```

**Trade-off**: 
- More memory usage (exponential with sequence length for quantum attention)
- Slower training
- BUT: Model can actually see the signal

### 2. **BETTER FIX: Fix Data Generation**

Modify `preprocess_genomics.py` to inject motifs in a **fixed position** (e.g., always in first 192bp):

```python
def inject_motif(sequence: str, position: int = 100) -> str:
    """Inject motif at a fixed position (e.g., position 100)."""
    motif = promoter_motifs[np.random.randint(0, len(promoter_motifs))]
    return sequence[:position] + motif + sequence[position + len(motif):]
```

This ensures **100% of positives have the signal in the truncated window**.

### 3. **BEST FIX: Use Real Genomic Data**

Replace synthetic labels with real promoter annotations from databases like:
- **ENSEMBL** gene annotations
- **NCBI RefSeq** promoter regions  
- **EPD** (Eukaryotic Promoter Database)

This eliminates label noise and provides biologically meaningful patterns.

### 4. **ALTERNATIVE: Reduce Model Capacity**

Since the task is simple (detect presence of 4 motifs), a simpler model might work better:

```bash
--qubits 2
--layers 1
--ansatz_layers 1
--embedding_dim 32
--dropout 0.4          # Very high dropout
--weight_decay 0.01    # Strong L2 regularization
```

But this won't fix the fundamental signal problem.

### 5. **NUCLEAR OPTION: Switch Task**

If this is for the hackathon demo:
- Use **lambeq embeddings** (already generated) with a **simple MLP classifier**
- The quantum circuit embeddings from lambeq are more meaningful than raw k-mer tokens
- Quixer wasn't designed for this use case anyway

## Recommended Next Steps

### Option A: Quick Fix for Demo (Highest Success Probability)

1. Increase sequence length to 128:
   ```bash
   CUDA_VISIBLE_DEVICES=0 bash run_quixer_training.sh \
     --max_seq_len 128 \
     --batch_size 16 \
     --qubits 4 \
     --layers 2 \
     --ansatz_layers 2 \
     --embedding_dim 64 \
     --dropout 0.3 \
     --weight_decay 0.005 \
     --lr 0.001 \
     --epochs 30
   ```

2. Expected results:
   - Val accuracy: ~60-65% (matches logistic regression baseline)
   - Still overfits, but learns *something*

### Option B: Proper Fix (Best Science)

1. Regenerate data with fixed motif positions:
   ```bash
   # Modify preprocess_genomics.py first
   python preprocess_genomics.py --n_samples 10000
   ```

2. Train with original Quixer config:
   ```bash
   CUDA_VISIBLE_DEVICES=0 bash run_quixer_training.sh \
     --max_seq_len 64 \
     --batch_size 32 \
     --qubits 6 \
     --layers 3 \
     --ansatz_layers 4 \
     --embedding_dim 256 \
     --epochs 50
   ```

3. Expected results:
   - Val accuracy: 70-80%
   - Proper learning curve

### Option C: Use Lambeq Embeddings (Most Aligned with Original Plan)

The original plan was to use **lambeq quantum embeddings**, not raw k-mer tokens. Switch back to that:

1. Generate lambeq embeddings:
   ```bash
   python lambeq_encoder.py --embedding_dim 64 --layers 2 --workers 4
   ```

2. Train on embeddings (modify training script to load `.pt` files instead of tokenizing)

3. This is more aligned with the "quantum genomics" story

## What I Recommend for Your Hackathon

Given the timeline pressure (due Friday), I recommend **Option A**:

1. Run with `--max_seq_len 128` immediately
2. Accept ~60% accuracy as a baseline
3. If it works, try Option B (regenerate data)
4. If time permits, pursue Option C (lambeq embeddings)

The key insight: **Your model can't learn what it can't see.** Fix the data pipeline first, then worry about hyperparameters.
