# Genomic Quixer Training (No Lambeq)

This guide explains how to train Quixer on RNA sequences without using Lambeq embeddings.

## Overview

Instead of the lambeq → quantization → Quixer pipeline, we now use:
1. **Tokenize** FASTA sequences using k-mers
2. **Train** Quixer directly on token IDs (like Penn Treebank)

## Quick Start

### Step 1: Build Dataset

Tokenize RNA sequences from FASTA file:

```bash
cd /scratch/cbjp404/bradford_hackathon_gene_sequencing

# Overlapping k-mers (default, more tokens)
python build_genomic_dataset.py \
    --fasta GRCh38_genomic_dataset/rna_sequences.fasta \
    --k 6 \
    --n_samples 10000 \
    --output genomic_data

# Non-overlapping k-mers (fewer tokens, no redundancy)
python build_genomic_dataset.py \
    --fasta GRCh38_genomic_dataset/rna_sequences.fasta \
    --k 6 \
    --n_samples 10000 \
    --no-overlap \
    --output genomic_data_nonoverlap
```

**What this does:**
- Reads `rna_sequences.fasta` (328k transcripts)
- Tokenizes using overlapping 6-mers (e.g., `ATCGAT` → `ATCGAT`, `TCGATC`, etc.)
- Creates balanced promoter/non-promoter labels
- Builds vocabulary with special tokens (`<pad>`, `<unk>`, `<eos>`)
- Splits into train/val/test (70/15/15)
- Saves to `genomic_data/`:
  - `vocab.json` - k-mer vocabulary
  - `train_tokens.pt` - training sequences (flattened)
  - `val_tokens.pt` - validation sequences
  - `test_tokens.pt` - test sequences
  - `metadata.json` - dataset statistics

**Expected output:**
```
Total transcripts: 328,868
Vocabulary size: ~4,096 tokens (6-mers)
Train tokens: ~70k
Val tokens: ~15k
Test tokens: ~15k
```

### Step 2: Train Quixer

Train Quixer on genomic data:

```bash
cd /scratch/cbjp404/Quixer_main

# Single GPU
python run.py --model Quixer --device cuda:0 --genomic

# Multi-GPU (3 GPUs)
torchrun --nproc_per_node=3 run.py --model Quixer --genomic
```

**Arguments:**
- `--genomic` - Use genomic dataset instead of Penn Treebank
- `--genomic-data-dir PATH` - Custom dataset path (default: `../bradford_hackathon_gene_sequencing/genomic_data`)
- `--model` - Model to train: `Quixer`, `Transformer`, `LSTM`, or `FNet`
- `--device` - Device (e.g., `cuda:0`, `cpu`)

### Step 3: Monitor Training

Training output shows:
```
Loading genomic dataset from /scratch/cbjp404/bradford_hackathon_gene_sequencing/genomic_data
  Vocabulary size: 4096
  Train tokens: 70,000
  Val tokens: 15,000
  Test tokens: 15,000
  Prepared batches: train=2187, val=468, test=468

Epoch: 01 | Time: 5m 23s
	Train Loss: 6.234 | Train ppl: 507.891
	 Val. Loss: 5.892 |  Val. ppl: 363.245

...

FINAL TRAINED MODEL STATS:
	 Val. Loss: 3.456 |  Val. ppl: 31.782
	 Test Loss: 3.521 |  Test ppl: 33.851
```

**Models saved to:** `Quixer_main/trained_models/q_transformer_lm_Quixer_{seed}_{timestamp}.pt`

## Comparison: Lambeq vs Classical Pipeline

| Component | Lambeq Pipeline | Classical Pipeline |
|-----------|----------------|-------------------|
| **Preprocessing** | `lambeq_encoder.py` (slow, compositional QNLP) | `build_genomic_dataset.py` (fast, k-mer tokenization) |
| **Embedding Init** | Cluster centroids from lambeq embeddings | Random (Xavier/normal) |
| **Vocab Size** | 512 (clustered) | ~4,096 (6-mers) or custom |
| **Training Input** | Quantized token IDs | Direct token IDs |
| **Time to Prep** | ~2-4 hours (GPU) | ~5 minutes (CPU) |
| **Disk Usage** | ~50 GB (chunks) | ~100 MB (tensors) |

## Dataset Details

### Tokenization Strategy

**Overlapping vs Non-overlapping K-mers:**

**Overlapping (stride=1, default):**
```
Sequence: ATCGATCG
K-mers:   ATCGAT, TCGATC, CGATCG
Tokens:   [1234, 5678, 9012]
Pros:     Dense representation, captures all positional information
Cons:     More tokens, potential information redundancy
```

**Non-overlapping (stride=k, use --no-overlap):**
```
Sequence: ATCGATCG
K-mers:   ATCGAT, CG (incomplete, discarded)
Tokens:   [1234]
Pros:     Fewer tokens, no redundancy, faster training
Cons:     May miss motifs at boundaries, less positional resolution
```

**When to use non-overlapping:**
- Your task requires distinct, non-redundant sequence units
- Training speed is critical (6× fewer tokens for k=6)
- You have shorter sequences and want to avoid over-representation
- Your downstream application processes sequences in fixed blocks

**When to use overlapping:**
- You need fine-grained positional information (e.g., splice site prediction)
- Motif detection is important (overlapping ensures no motif is split)
- You're following DNABERT/Nucleotide Transformer conventions
- You have sufficient compute resources

**Why k=6?**
- Balances vocab size (4^6 = 4,096) vs context
- Captures promoter motifs (TATAAA, TTGACA, CCAAT)
- Matches original pipeline window size expectations

**Special tokens:**
- `<pad>` (ID 0): Padding for shorter sequences
- `<unk>` (ID 1): Unknown k-mers (low frequency)
- `<eos>` (ID 2): End of sequence marker

### Labeling Strategy

**Synthetic promoter/non-promoter labels:**
1. Check FASTA headers for keywords: `promoter`, `transcript`, `mrna`, `gene`
2. Check sequences for motifs: `TATAAA`, `TTGACA`, `CCAAT`, `GGGCGG`
3. Label = 1 if keywords OR motifs present, else 0
4. Balance classes 1:1 to avoid degenerate training

**Example:**
```
>ENST00000622028.1 ... transcript_biotype:IG_V_gene gene_symbol:IGHV1OR21-1
ATGGACTGGAATTGGAGGATCCTGTTT...TATAAA...GCAGA
→ Label: 1 (promoter - has keyword "transcript" and motif "TATAAA")
```

## Customization

### Change K-mer Size

Use `--k` flag to change k-mer size:

```bash
# 4-mers (vocab size = 256)
python build_genomic_dataset.py --k 4 --output genomic_data_k4

# 8-mers (vocab size = 65,536)
python build_genomic_dataset.py --k 8 --output genomic_data_k8
```

**Trade-offs:**
- **Smaller k** (3-4): Smaller vocab, less context per token, faster training
- **Larger k** (8-10): Larger vocab, more context per token, slower training

### Use Custom FASTA

Point to your own FASTA file:

```bash
python build_genomic_dataset.py \
    --fasta /path/to/your/sequences.fasta \
    --output custom_genomic_data
```

Then train:

```bash
python run.py --genomic --genomic-data-dir custom_genomic_data
```

### Adjust Vocab Filtering

Control minimum k-mer frequency:

```bash
# Only include k-mers appearing ≥5 times
python build_genomic_dataset.py --min_freq 5
```

This reduces vocab size and filters noise.

## Hyperparameters

Default Quixer hyperparameters (from `run.py`):

```python
quixer_hparams = {
    "qubits": 6,
    "layers": 3,
    "ansatz_layers": 4,
    "window": 32,           # Context window size
    "epochs": 30,
    "dropout": 0.10,
    "lr": 0.002,
    "lr_sched": "cos",      # Cosine annealing
    "wd": 0.0001,
    "batch_size": 32,
    "max_grad_norm": 5.0,
    "dimension": 512,       # Embedding dimension
}
```

These match the Penn Treebank baseline and are suitable for genomic data.

## Expected Results

### Penn Treebank (Text)
- **Test perplexity**: ~100-150
- **Task**: Next-token prediction on natural language

### Genomic (RNA sequences)
- **Test perplexity**: ~30-50 (expected)
- **Task**: Next k-mer prediction + promoter classification
- **Reason**: Genomic sequences have more regular patterns than natural language

**Note:** Lower perplexity ≠ better biology. The model learns statistical patterns but may not capture functional relationships without domain-specific pretraining.

## Troubleshooting

### Out of Memory

**Reduce batch size:**
```python
# In run.py, modify:
quixer_hparams["batch_size"] = 16  # or 8
```

### Vocabulary Too Large

**Increase min_freq:**
```bash
python build_genomic_dataset.py --min_freq 5
```

### Training Diverges

**Lower learning rate:**
```python
# In run.py, modify:
quixer_hparams["lr"] = 0.001  # or 0.0005
```

## Next Steps

1. **Evaluate on real tasks**: Replace synthetic labels with real annotations (e.g., promoter databases)
2. **Try classical baselines**: Compare with `--model Transformer`, `LSTM`, `FNet`
3. **Tune hyperparameters**: Adjust learning rate, dropout, layers
4. **Scale up**: Use full FASTA (328k sequences) instead of subset
5. **Multi-task learning**: Add splice site, enhancer, or regulatory element prediction

## Files Overview

```
bradford_hackathon_gene_sequencing/
├── build_genomic_dataset.py        # Tokenize FASTA → dataset
├── genomic_data/                    # Output directory
│   ├── vocab.json
│   ├── train_tokens.pt
│   ├── val_tokens.pt
│   ├── test_tokens.pt
│   └── metadata.json
├── Quixer/quixer/setup_training.py # Modified to support genomic data
└── GRCh38_genomic_dataset/
    └── rna_sequences.fasta         # Input sequences

Quixer_main/
├── run.py                          # Modified to add --genomic flag
└── trained_models/                 # Output checkpoints
```

## References

- **Quixer paper**: arXiv:2406.04305 (LCU + QSVT quantum transformer)
- **Tokenization study**: "Effect of tokenization on transformers for biological sequences" (PMC11055402)
- **Penn Treebank**: Original dataset for language modeling benchmarks
- **GRCh38**: Human reference genome assembly

---

**Ready to train!** Run:
```bash
python build_genomic_dataset.py
cd ../Quixer_main
python run.py --model Quixer --device cuda:0 --genomic
```
