# Genomics Quantum NLP Pipeline Architecture

## Overview

This document describes the complete quantum-classical hybrid pipeline for genomic sequence classification using lambeq compositional encodings and Quixer quantum transformers.

```
================================================================================
                    GENOMICS QUANTUM NLP PIPELINE ARCHITECTURE
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAW DATA INPUT                                     │
│                                                                              │
│  GRCh38_genomic_dataset/                                                    │
│    └── GRCh38_latest_rna_summary.csv  (RNA transcript summaries)           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: PREPROCESSING (preprocess_genomics.py)                            │
│  ────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  GenomicDataPreprocessor:                                                   │
│    ├─ Load RNA data (CSV)                                                  │
│    ├─ Create synthetic labels (promoter vs non-promoter)                    │
│    ├─ Extract k-mers (6-mer tokenization)                                  │
│    ├─ Convert sequences → k-mer "sentences"                                │
│    ├─ Build vocabulary (k-mer → index mapping)                             │
│    └─ Split data (70/15/15 train/val/test)                                 │
│                                                                              │
│  Output: processed_data/                                                    │
│    ├── train/                                                              │
│    │   ├── sentences.txt  (k-mer sentences, one per line)                  │
│    │   └── labels.txt     (0/1 binary labels)                              │
│    ├── val/                                                                 │
│    ├── test/                                                                │
│    ├── vocab.json          (k-mer vocabulary mapping)                      │
│    └── metadata.json       (k, window_size, n_classes, etc.)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: QUANTUM ENCODING (lambeq_encoder.py)                              │
│  ────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  GenomicLambeqEncoder:                                                      │
│    ├─ Parse k-mer sentences → DisCoCat diagrams (BobcatParser)             │
│    ├─ Convert diagrams → quantum circuits (IQP ansatz)                    │
│    ├─ Simulate circuits → per-k-mer embeddings (64-dim each)               │
│    ├─ Preserve positional structure (sequences of embeddings)              │
│    └─ Export embeddings for each split                                       │
│                                                                              │
│  Key Innovation: Per-K-mer Embeddings                                       │
│    ├─ Instead of one embedding per sentence                                │
│    ├─ Generate one embedding per k-mer token                               │
│    ├─ Preserve sequence structure for Quixer attention                     │
│    └─ Output shape: [N_samples, max_kmers, 64]                             │
│                                                                              │
│  Fallback: SimpleKmerEncoder (if lambeq unavailable)                        │
│    └─ Mean pooling of k-mer embeddings                                      │
│                                                                              │
│  Output: lambeq_embeddings/                                                 │
│    ├── train.pt      (PyTorch tensors: per-kmer embeddings + labels)       │
│    ├── val.pt                                                                │
│    ├── test.pt                                                              │
│    └── encoding_metadata.json                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: VECTOR QUANTIZATION (quantize_lambeq_embeddings.py)               │
│  ────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  Quantization Process:                                                      │
│    ├─ Load per-k-mer embeddings [N, max_kmers, 64]                         │
│    ├─ Flatten to [N*max_kmers, 64] for k-means clustering                  │
│    ├─ Fit MiniBatchKMeans (512 clusters)                                    │
│    ├─ Assign each embedding to nearest centroid                             │
│    ├─ Create token sequences [N, seq_len] with token IDs                   │
│    ├─ Truncate/pad to fixed sequence length (32 tokens)                    │
│    └─ Save cluster centroids for embedding initialization                   │
│                                                                              │
│  Why Quantization?                                                          │
│    ├─ Reduces continuous embeddings to discrete vocabulary                 │
│    ├─ Enables efficient token-based processing in Quixer                   │
│    ├─ Preserves semantic structure via cluster centroids                   │
│    └─ Centroids initialize Quixer embedding layer                          │
│                                                                              │
│  Output: quantized_embeddings/                                              │
│    ├── train.pt              (token sequences + labels)                     │
│    ├── val.pt                                                                │
│    ├── test.pt                                                              │
│    ├── cluster_centers.pt    (512 × 64 centroid matrix)                    │
│    ├── kmeans_model.pkl      (fitted k-means for inference)                │
│    └── metadata.json         (n_clusters, seq_len, embedding_dim, etc.)    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: QUIXER TRAINING (train_quixer_hybrid.py)                          │
│  ────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  Model Initialization:                                                      │
│    ├─ Load cluster centroids from quantized_embeddings/                    │
│    ├─ Create QuixerClassifier with:                                        │
│    │   ├─ n_qubits: 4-8 (configurable)                                     │
│    │   ├─ n_tokens: 32 (sequence length)                                   │
│    │   ├─ vocabulary_size: 512 (k-means clusters)                          │
│    │   └─ embedding_dimension: 64 (from metadata)                          │
│    ├─ Initialize embedding layer with cluster centroids                    │
│    └─ Freeze or fine-tune based on config                                  │
│                                                                              │
│  Training Loop:                                                             │
│    ├─ Load quantized token sequences                                        │
│    ├─ Forward pass:                                                         │
│    │   ├─ Token IDs → Embedding layer (initialized with centroids)        │
│    │   ├─ Embeddings → Linear layer (WE) → PQC angles                     │
│    │   ├─ Angles → Parameterized quantum circuits                         │
│    │   ├─ LCU + QSVT operations on quantum state                          │
│    │   ├─ Measure expectation values (X, Y, Z)                            │
│    │   └─ Classification head → logits                                      │
│    ├─ Compute loss (CrossEntropyLoss)                                      │
│    ├─ Backprop through classical components                                │
│    └─ Update parameters (AdamW optimizer)                                  │
│                                                                              │
│  Hyperparameters:                                                           │
│    ├─ Learning rate: 1e-3 to 5e-3                                          │
│    ├─ Batch size: 32-64                                                    │
│    ├─ Epochs: 50-100                                                       │
│    ├─ Dropout: 0.1-0.3                                                     │
│    └─ Scheduler: CosineAnnealingWarmRestarts or StepLR                     │
│                                                                              │
│  Output: quixer_hybrid_results/                                             │
│    ├── model_checkpoint.pt   (trained model weights)                       │
│    ├── training_history.json (loss/accuracy curves)                        │
│    ├── metrics.json          (test accuracy, F1, confusion matrix)         │
│    └── logs/                 (training logs)                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: HYPERPARAMETER TUNING (tune_quixer_hybrid.py)                     │
│  ────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  Optuna Framework:                                                          │
│    ├─ Objective: Maximize validation accuracy                              │
│    ├─ Sampled parameters:                                                  │
│    │   ├─ n_qubits: [4, 5, 6, 7, 8]                                        │
│    │   ├─ n_layers: [2, 3, 4, 5]                                           │
│    │   ├─ ansatz_layers: [2, 3, 4, 5, 6]                                   │
│    │   ├─ dropout: [0.05, 0.4]                                             │
│    │   ├─ learning_rate: [5e-4, 5e-3] (log scale)                         │
│    │   ├─ weight_decay: [1e-5, 1e-2] (log scale)                          │
│    │   └─ batch_size: [16, 32, 48, 64]                                     │
│    ├─ Fixed parameters (from metadata):                                    │
│    │   ├─ embedding_dim: 64                                                │
│    │   ├─ vocabulary_size: 512                                             │
│    │   └─ seq_len: 32                                                      │
│    ├─ Pruning: Optuna pruner for early stopping                            │
│    └─ Trials: 50-100 (configurable)                                        │
│                                                                              │
│  Output: optuna_results/                                                    │
│    ├── study.db              (Optuna study database)                       │
│    ├── best_params.json      (optimal hyperparameters)                     │
│    ├── trial_history.csv     (all trial results)                           │
│    └── optimization_plots.png (Optuna visualization)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  EVALUATION & COMPARISON                                                     │
│  ────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  Metrics Computed:                                                          │
│    ├─ Accuracy (train/val/test)                                             │
│    ├─ F1-score (weighted, per-class)                                        │
│    ├─ Confusion matrix                                                      │
│    ├─ Precision, Recall, Specificity                                        │
│    ├─ Parameter count                                                       │
│    ├─ Training time                                                         │
│    └─ Inference latency                                                     │
│                                                                              │
│  Comparison Baselines:                                                      │
│    ├─ Classical LSTM (from run_genomics_training.py)                       │
│    ├─ Classical Transformer (from run_genomics_training.py)                │
│    └─ Random classifier (50% baseline)                                      │
│                                                                              │
│  Success Criteria:                                                          │
│    ✅ Quixer test accuracy ≥ 80%                                            │
│    ✅ Quixer within ±3% of classical baselines                              │
│    ✅ Quixer uses ≤50% parameters vs. Transformer                           │
│    ✅ F1-score > 0.80 on balanced test set                                  │
│    ✅ Training converges within 50 epochs                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Innovations

### 1. Per-K-mer Embeddings
**Problem**: Original pipeline generated one embedding per entire sequence, losing positional structure.

**Solution**: Modified `lambeq_encoder.py` to generate one embedding per k-mer token:
- Each k-mer in the sequence gets its own 64-dimensional embedding
- Preserves sequential structure for Quixer's attention mechanism
- Output: `[N_samples, max_kmers, 64]` instead of `[N_samples, 64]`

### 2. Vector Quantization with Centroid Initialization
**Problem**: Quixer's embedding layer started with random weights, no connection to lambeq encodings.

**Solution**: Implemented quantization pipeline:
- Cluster per-k-mer embeddings using MiniBatchKMeans (512 clusters)
- Save cluster centroids as `cluster_centers.pt`
- Initialize Quixer's embedding layer with these centroids
- Token IDs now map to meaningful lambeq-derived vectors

### 3. Hybrid Classical-Quantum Processing
**Architecture**:
```
Quantized Token Sequences (discrete)
    ↓
Embedding Layer (initialized with lambeq centroids)
    ↓
Classical Linear Layer (WE: embedding_dim → n_pqc_parameters)
    ↓
Parameterized Quantum Circuits (IQP ansatz)
    ↓
LCU + QSVT Operations (quantum attention)
    ↓
Measurement & Expectation Values
    ↓
Classical Classification Head
    ↓
Binary Prediction (promoter/non-promoter)
```

## Data Flow

### Input
- Raw genomic sequences from GRCh38 RNA summary CSV
- 512 bp windows, labeled as promoter/non-promoter

### Processing
1. **Preprocessing**: Extract k-mers, create sentences, split data
2. **Encoding**: Generate per-k-mer quantum embeddings via lambeq
3. **Quantization**: Cluster embeddings, create token sequences
4. **Training**: Train Quixer with initialized embeddings
5. **Tuning**: Optimize hyperparameters with Optuna

### Output
- Trained Quixer model
- Classification metrics (accuracy, F1, confusion matrix)
- Comparison with classical baselines
- Hyperparameter optimization results

## File Organization

```
bradford_hackathon_2025/
├── ARCHITECTURE.md                    # This file
├── README.md                          # Quick start guide
├── PER_KMER_PIPELINE.md              # Detailed per-k-mer changes
├── QUIXER_HYPERPARAMETERS.md         # Tuning guide
│
├── preprocess_genomics.py            # Stage 1: Preprocessing
├── lambeq_encoder.py                 # Stage 2: Quantum encoding (per-k-mer)
├── quantize_lambeq_embeddings.py     # Stage 3: Vector quantization
├── train_quixer_hybrid.py            # Stage 4: Training
├── tune_quixer_hybrid.py             # Stage 5: Hyperparameter tuning
│
├── processed_data/                   # Stage 1 output
├── lambeq_embeddings/                # Stage 2 output (per-k-mer)
├── quantized_embeddings/             # Stage 3 output
│   ├── train.pt
│   ├── val.pt
│   ├── test.pt
│   ├── cluster_centers.pt            # Centroid initialization
│   ├── kmeans_model.pkl
│   └── metadata.json
├── quixer_hybrid_results/            # Stage 4 output
├── optuna_results/                   # Stage 5 output
│
└── Quixer/                           # Quantum transformer implementation
    ├── quixer/
    │   ├── quixer_model.py           # Core Quixer model
    │   ├── quixer_classifier.py      # Classification wrapper
    │   └── ...
    └── ...
```

## Running the Pipeline

### Full Pipeline (Recommended)
```bash
cd /scratch/cbjp404/bradford_hackathon_2025
bash run_hybrid_pipeline.sh
```

### Step-by-Step
```bash
# 1. Preprocess
python preprocess_genomics.py

# 2. Generate per-k-mer embeddings
python lambeq_encoder.py

# 3. Quantize and save centroids
python quantize_lambeq_embeddings.py --n_clusters 512 --seq_len 32

# 4. Train Quixer with initialized embeddings
python train_quixer_hybrid.py --qubits 6 --layers 3 --ansatz_layers 4 --epochs 50

# 5. Tune hyperparameters
python tune_quixer_hybrid.py --n_trials 50 --epochs_per_trial 10
```

## References

1. **Quixer**: arXiv:2406.04305 - Quantum Transformer with LCU+QSVT
2. **lambeq**: https://docs.quantinuum.com/lambeq/ - Compositional QNLP
3. **GRCh38**: NCBI Reference Genome
4. **iMOKA**: Genome Biology (2020) - k-mer ML for genomics
5. **DNABERT-2**: arXiv:2306.15006 - Transformer benchmarks

## Contact

Primary: Yana (YANAGPU)  
Collaborator: Sid  
Location: Leeds AIRE HPC  
GPUs: 2× NVIDIA L40 (48 GB each)
