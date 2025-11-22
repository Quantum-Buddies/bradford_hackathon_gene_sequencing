# Quixer: Quantum-Enhanced Genomics

Quixer is a hybrid Quantum-Classical Transformer designed for parameter-efficient genomic sequence analysis. By leveraging quantum state space mappings (via `lambeq` and `QSVT`), Quixer achieves state-of-the-art (SOTA) comparable accuracy with **100x fewer parameters** than classical baselines like DNABERT.

## Key Advantages
*   **Efficiency:** ~136k parameters vs 12M+ for classical models.
*   **Scalability:** Linear $O(N)$ scaling vs $O(N^2)$ for classical attention.
*   **Trainability:** Solves the "Barren Plateau" problem via Hybrid Centroid Initialization.
*   **Biological Insight:** Interpretable attention maps that identify regulatory motifs.

## Project Structure
```
.
├── src/
│   ├── training/   # Training scripts (Classical Baseline & Quixer)
│   ├── inference/  # Inference & Validation scripts
│   ├── utils/      # Plotting and helper functions
│   └── models/     # Model definitions
├── models/         # Saved model checkpoints (.pt)
├── plots/          # Generated figures for analysis/pitching
├── results/        # Logs and metrics
└── notebooks/      # Jupyter notebooks for exploration
```

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Classical Baseline Training
To reproduce the classical benchmark comparison:
```bash
python src/training/train_baseline.py
```
This generates `plots/kmer_baseline_validation.png`.

### 3. Run Quixer Inference
To evaluate the pre-trained Quixer model:
```bash
python src/inference/inference_quixer.py
```
This generates `plots/nextbase_confusion_matrix_real.png`.

### 4. Generate Investor Pitch Assets
To create the full suite of efficiency and scalability charts:
```bash
python src/utils/generate_plots.py
```

## Hardware Requirements
*   **GPU:** NVIDIA L40 / A100 recommended (runs on single GPU).
*   **RAM:** 16GB+

## Contact
Bradford Hackathon Team
