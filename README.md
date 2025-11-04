# Bradford Hackathon 2025 – Quantum Genomics

Minimal instructions for reproducing the lambeq + Quixer genomics pipeline.

## Prerequisites

- Linux environment with Conda
- Access to the GRCh38 CSV bundle (see below)
- GPUs optional (L40 preferred)

### Suggested Conda env
```bash
module load miniforge
conda create -n bradford python=3.11
conda activate bradford
pip install lambeq[extras] torch scikit-learn pandas tqdm
```

## Repository layout

```
├── preprocess_genomics.py   # build k-mer sentences + labels
├── lambeq_encoder.py        # lambeq Bobcat + IQP embeddings (fallback provided)
├── run_genomics_training.py # train Quixer/LSTM/Transformer on embeddings
├── run_genomics_quixer.sh   # Slurm wrapper for end-to-end run
├── check_setup.py           # sanity checks before launching jobs
├── Quixer/                  # training code (minified copy of upstream)
├── GRCh38_genomic_dataset/  # raw CSVs (not tracked in gitignore)
└── results/, logs/, …       # generated artefacts
```

## Data preparation

1. Place the GRCh38 summaries under `GRCh38_genomic_dataset/`.
2. Run preprocessing:
   ```bash
   python preprocess_genomics.py
   ```
   Outputs go to `processed_data/` (train/val/test splits + vocab).

## Encoding (lambeq IQP ansatz)

```bash
python lambeq_encoder.py \
  --embedding_dim 512 \
  --n_layers 2
```

- Requires Bobcat model download on first run.
- Fallback encoder writes to `simple_embeddings/` if lambeq fails.

## Training

```bash
python run_genomics_training.py \
  --models Quixer LSTM Transformer \
  --embeddings_dir lambeq_embeddings \
  --epochs 50 \
  --batch_size 32
```

Slurm submission (two L40s):
```bash
sbatch run_genomics_quixer.sh
```

Artifacts land in `results/` (JSON metrics, confusion matrices) and `logs/`.

## Quick checklist

- [ ] `python check_setup.py`
- [ ] `python preprocess_genomics.py`
- [ ] `python lambeq_encoder.py`
- [ ] `python run_genomics_training.py`
- [ ] Inspect `results/*.json`

Keep large CSVs and generated tensors out of git; see `.gitignore` for the default exclusions.
