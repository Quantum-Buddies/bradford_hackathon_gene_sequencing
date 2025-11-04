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
[2. lambeq Encoding]
      ↓
Quantum Circuit Embeddings (512-dim)
      ↓
[3. Classification Models]
   ├── Quixer (Quantum Transformer)
   ├── LSTM (Classical Baseline)
   └── Transformer (Classical Baseline)
      ↓
Binary Classification: Promoter vs. Non-Promoter
```

## Pipeline Components

### 1. Data Preprocessing (`preprocess_genomics.py`)
- Extracts 512 bp windows from GRCh38 genomic/RNA summaries
- Tokenizes sequences into overlapping 6-mers
- Labels regions as promoter/non-promoter based on annotations
- Generates train/val/test splits (70/15/15)
- Exports k-mer "sentences" for lambeq

**Output**: `/scratch/cbjp404/bradford_hackathon_2025/processed_data/`

### 2. lambeq Encoding (`lambeq_encoder.py`)
- Parses k-mer sentences using DisCoCat compositional grammar
- Applies IQP ansatz to generate parameterized quantum circuits
- Simulates circuits to produce 512-dimensional embeddings
- Exports PyTorch tensors for training

**Output**: `/scratch/cbjp404/bradford_hackathon_2025/lambeq_embeddings/`

### 3. Model Training (`run_genomics_training.py`)
- Loads lambeq embeddings
- Trains multiple models:
  - **Quixer**: Quantum attention (simulated on classical hardware)
  - **LSTM**: Recurrent baseline
  - **Transformer**: Attention baseline
- Evaluates accuracy, F1-score, parameter count
- Generates comprehensive reports

**Output**: `/scratch/cbjp404/bradford_hackathon_2025/results/`

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

### Option 1: Full Pipeline (Slurm on L40s)
```bash
cd /scratch/cbjp404/bradford_hackathon_2025
chmod +x run_genomics_quixer.sh
sbatch run_genomics_quixer.sh
```

This will:
1. Preprocess data (if needed)
2. Generate lambeq embeddings (if needed)
3. Train all models
4. Save results

### Option 2: Step-by-Step Execution

**Step 1: Preprocess**
```bash
python preprocess_genomics.py
```

**Step 2: Encode with lambeq**
```bash
python lambeq_encoder.py
```

**Step 3: Train Models**
```bash
python run_genomics_training.py \
    --models Quixer LSTM Transformer \
    --device cuda \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001
```

### Option 3: Interactive (Single Model)
```bash
python run_genomics_training.py \
    --models Quixer \
    --device cuda \
    --epochs 30
```

## Expected Results

### Classical Baselines (from literature)
- **iMOKA (Random Forest)**: ~95% accuracy on breast cancer RNA-Seq subtyping
- **DNABERT-2 (Transformer)**: State-of-the-art on GUE benchmark tasks
- **LSTM**: 85-90% on regulatory element prediction

### Our Targets
| Model | Expected Test Acc | Parameters | Notes |
|-------|------------------|------------|-------|
| **LSTM** | 80-85% | ~500K | Classical recurrent baseline |
| **Transformer** | 85-90% | ~1-2M | Classical attention baseline |
| **Quixer** | 83-88% | **<500K** | **Quantum attention, fewer params** |

### Success Criteria
✅ Quixer matches classical accuracy ±3%  
✅ Quixer uses ≤50% parameters vs. Transformer  
✅ Training converges within 50 epochs  
✅ F1-score >0.80 on balanced test set

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
