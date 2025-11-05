# Bradford Hackathon 2025: Quantum Genomics with lambeq + Quixer

## Overview

This project demonstrates quantum-enhanced genomic sequence classification using:
- **lambeq**: Compositional QNLP framework for encoding genomic k-mer sequences
- **Quixer**: Quantum transformer with LCU+QSVT attention mechanisms
- **GRCh38 Dataset**: Human genome reference sequences

### Research Question
Can lambeq's compositional encodings + Quixer's quantum attention achieve comparable classification accuracy with fewer trainable parameters than classical transformers/LSTMs?

## Architecture

```
GRCh38 CSV Files
      ↓
[1. Preprocessing]
      ↓
K-mer Tokenization (6-mers, 512 bp windows)
      ↓
[2. lambeq Encoding] ← Per-K-mer Embeddings (NEW)
      ↓
Quantum Circuit Embeddings (64-dim per k-mer, preserves sequence structure)
      ↓
[3. Vector Quantization] ← Cluster Centroids Initialization (NEW)
      ↓
Discrete Token Sequences + Centroid-Initialized Embeddings
      ↓
[4. Quixer Training]
      ↓
Binary Classification: Promoter vs. Non-Promoter
```

**See `ARCHITECTURE.md` for detailed pipeline diagram and component descriptions.**

## Pipeline Components

### 1. Data Preprocessing (`preprocess_genomics.py`)
- Extracts 512 bp windows from GRCh38 genomic/RNA summaries
- Tokenizes sequences into overlapping 6-mers
- Labels regions as promoter/non-promoter based on annotations
- Generates train/val/test splits (70/15/15)
- Exports k-mer "sentences" for lambeq

**Output**: `/scratch/cbjp404/bradford_hackathon_2025/processed_data/`

### 2. lambeq Encoding (`lambeq_encoder.py`) — Per-K-mer Embeddings
- Parses k-mer sentences using DisCoCat compositional grammar
- Applies IQP ansatz to generate parameterized quantum circuits
- **NEW**: Generates one embedding per k-mer token (64-dim each)
- **NEW**: Preserves sequence structure: `[N_samples, max_kmers, 64]`
- Exports PyTorch tensors for training

**Key Innovation**: Per-k-mer embeddings preserve positional information, enabling Quixer's quantum attention to operate on diverse token representations instead of collapsed sequences.

**Output**: `/scratch/cbjp404/bradford_hackathon_2025/lambeq_embeddings/`

### 3. Vector Quantization (`quantize_lambeq_embeddings.py`) — Centroid Initialization
- Loads per-k-mer embeddings from Stage 2
- Clusters embeddings using MiniBatchKMeans (512 clusters)
- Creates discrete token sequences: `[N_samples, seq_len]`
- **NEW**: Saves cluster centroids for embedding layer initialization
- Validates cluster utilization and token diversity

**Key Innovation**: Cluster centroids initialize Quixer's embedding layer, connecting discrete tokens to meaningful lambeq-derived vectors.

**Output**: `/scratch/cbjp404/bradford_hackathon_2025/quantized_embeddings/`
- `train.pt`, `val.pt`, `test.pt` (token sequences + labels)
- `cluster_centers.pt` (512 × 64 centroid matrix)
- `metadata.json` (n_clusters, seq_len, embedding_dim)

### 4. Quixer Training (`train_quixer_hybrid.py`)
- Loads cluster centroids from Stage 3
- Initializes QuixerClassifier embedding layer with centroids
- Trains quantum transformer with meaningful token representations
- Evaluates accuracy, F1-score, parameter count
- Generates comprehensive reports

**Output**: `/scratch/cbjp404/bradford_hackathon_2025/quixer_hybrid_results/`

### 5. Hyperparameter Tuning (`tune_quixer_hybrid.py`)
- Uses Optuna framework to optimize hyperparameters
- Samples: n_qubits, n_layers, ansatz_layers, dropout, lr, weight_decay, batch_size
- Fixed from metadata: embedding_dim, vocabulary_size, seq_len
- Maximizes validation accuracy with early pruning
- Generates optimization plots and best parameters

**Output**: `/scratch/cbjp404/bradford_hackathon_2025/optuna_results/`

## Setup

### Environment
```bash
# Activate Quixer conda environment
conda activate quixer

# Install additional dependencies
pip install lambeq scikit-learn pandas tqdm
```

### Data
Genomic CSV files are located at:
```
/scratch/cbjp404/bradford_hackathon_2025/GRCh38_genomic_dataset/
├── GRCh38_latest_genomic_summary.csv
├── GRCh38_latest_rna_summary.csv
└── GRCh38_latest_protein_symmery.csv
```

## Running the Pipeline

### Option 1: Full Hybrid Pipeline (Recommended)
```bash
cd /scratch/cbjp404/bradford_hackathon_2025
bash run_hybrid_pipeline.sh
```

This will:
1. Preprocess data (if needed)
2. Generate per-k-mer lambeq embeddings
3. Quantize embeddings and save cluster centroids
4. Train Quixer with centroid-initialized embeddings
5. Optionally tune hyperparameters with Optuna
6. Save all results

### Option 2: Step-by-Step Execution (Hybrid Pipeline)

**Step 1: Preprocess**
```bash
python preprocess_genomics.py
```

**Step 2: Generate Per-K-mer Embeddings**
```bash
python lambeq_encoder.py
```

**Step 3: Quantize and Save Centroids**
```bash
python quantize_lambeq_embeddings.py \
    --n_clusters 512 \
    --seq_len 32
```

**Step 4: Train Quixer with Initialized Embeddings**
```bash
python train_quixer_hybrid.py \
    --qubits 6 \
    --layers 3 \
    --ansatz_layers 4 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001
```

**Step 5: Tune Hyperparameters (Optional)**
```bash
python tune_quixer_hybrid.py \
    --n_trials 50 \
    --epochs_per_trial 10
```

### Option 3: Classical Baselines (for Comparison)
```bash
python run_genomics_training.py \
    --models LSTM Transformer \
    --device cuda \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001
```

### Option 4: Interactive (Single Model)
```bash
python train_quixer_hybrid.py \
    --qubits 6 \
    --layers 3 \
    --ansatz_layers 4 \
    --epochs 30 \
    --device cuda
```

## Expected Results

### Classical Baselines (from literature)
- **iMOKA (Random Forest)**: ~95% accuracy on breast cancer RNA-Seq subtyping
- **DNABERT-2 (Transformer)**: State-of-the-art on GUE benchmark tasks
- **LSTM**: 85-90% on regulatory element prediction

### Our Targets (with Per-K-mer + Centroid Initialization)
| Model | Expected Test Acc | Parameters | Notes |
|-------|------------------|------------|-------|
| **LSTM** | 80-85% | ~500K | Classical recurrent baseline |
| **Transformer** | 85-90% | ~1-2M | Classical attention baseline |
| **Quixer (Hybrid)** | **≥80%** | **<500K** | **Quantum attention, lambeq-initialized embeddings** |

### Success Criteria (Hybrid Pipeline)
✅ Quixer test accuracy ≥ 80% (improvement from ~50% random baseline)  
✅ Quixer within ±3% of classical baselines  
✅ Quixer uses ≤50% parameters vs. Transformer  
✅ Training converges within 50 epochs  
✅ F1-score > 0.80 on balanced test set  
✅ Per-k-mer embeddings preserve sequence structure  
✅ Cluster centroids properly initialize embedding layer

## Evaluation Metrics

- **Accuracy**: Overall classification correctness
- **F1-Score**: Weighted average (handles class imbalance)
- **Confusion Matrix**: Per-class performance breakdown
- **Parameter Count**: Model complexity comparison
- **Training Time**: Computational efficiency

## Outputs

### Results Directory Structure
```
/scratch/cbjp404/bradford_hackathon_2025/results/
├── results_YYYYMMDD_HHMMSS.json  # Full training history
├── plots/
│   ├── accuracy_comparison.png
│   ├── parameter_efficiency.png
│   └── confusion_matrices.png
└── logs/
    └── training_log.txt
```

### JSON Results Format
```json
{
  "Quixer": {
    "test_acc": 85.3,
    "test_f1": 0.8472,
    "n_parameters": 487235,
    "train_time": 234.5,
    "confusion_matrix": [[450, 50], [45, 455]],
    "history": {...}
  },
  ...
}
```

## References

### Key Papers
1. **QNLP in Bioinformatics**: Frontiers in Computer Science (2025)  
   - Demonstrated quantum embedding advantages for genomic sequence analysis

2. **Quixer**: arXiv:2406.04305  
   - Quantum transformer with LCU+QSVT primitives

3. **iMOKA**: Genome Biology (2020)  
   - k-mer based ML for human genomics classification

4. **DNABERT-2**: arXiv:2306.15006  
   - Foundation model benchmark on human genome tasks

### Tools
- **lambeq**: https://docs.quantinuum.com/lambeq/
- **Quixer**: https://github.com/Ryukijano/Quixer
- **GRCh38**: NCBI Reference Genome

## Team Division

### Yana (You)
- lambeq encoding pipeline
- Quixer integration
- Quantum experiments
- Results analysis

### Sid
- Classical baselines (LSTM, Transformer)
- Dataset curation & validation
- Metrics reporting
- Documentation

## Troubleshooting

### lambeq Import Error
```bash
pip install lambeq
# If unavailable, fallback encoder will use simple k-mer embeddings
```

### GPU Memory Issues
Reduce batch size:
```bash
python run_genomics_training.py --batch_size 16
```

### Slurm Job Failed
Check logs:
```bash
cat /scratch/cbjp404/bradford_hackathon_2025/logs/quixer_*.err
```

## Timeline

**Friday (Due Date)**
- [x] Data preprocessing complete
- [x] lambeq encoding pipeline ready
- [x] Training infrastructure built
- [ ] Run full experiments on L40s
- [ ] Generate comparison plots
- [ ] Prepare presentation deck

## Contact

Primary: Yana (YANAGPU)  
Collaborator: Sid  
Location: Leeds AIRE HPC  
GPUs: 2× NVIDIA L40 (48 GB each)

---

**Last Updated**: 2025-11-03  
**Status**: Ready for execution 🚀
