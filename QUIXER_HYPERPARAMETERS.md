# Quixer Genomics Training: Recommended Hyperparameters

Based on the original Quixer paper (arXiv:2406.04305) and the official implementation, here are the recommended hyperparameters for training on your genomics classification task.

## **Optimal Configuration (Based on PTB Experiments)**

### Core Model Parameters
```bash
--qubits 6                    # Number of qubits (paper used 6)
--layers 3                    # QSVT polynomial degree (cubic polynomial)
--ansatz_layers 4             # Layers in parameterized quantum circuit
--embedding_dim 512           # Token embedding dimension (paper used 512 for Quixer)
--max_seq_len 32              # Context window / sequence length
```

### Training Parameters
```bash
--batch_size 32               # Batch size (paper: 32 contexts × 32 tokens = 1024 effective)
--epochs 30                   # Number of training epochs
--lr 0.002                    # Learning rate (Adam optimizer)
--weight_decay 0.0001         # L2 regularization
--dropout 0.10                # Dropout rate (10%)
--scheduler cosine            # Cosine annealing LR schedule
```

### System Parameters
```bash
--device cuda                 # Use GPU for training
--seed 42                     # Random seed for reproducibility
```

---

## **Full Training Command (Optimal)**

```bash
CUDA_VISIBLE_DEVICES=0 bash run_quixer_training.sh \
  --qubits 6 \
  --layers 3 \
  --ansatz_layers 4 \
  --embedding_dim 512 \
  --max_seq_len 32 \
  --batch_size 32 \
  --epochs 30 \
  --lr 0.002 \
  --weight_decay 0.0001 \
  --dropout 0.10 \
  --scheduler cosine \
  --device cuda \
  --seed 42
```

---

## **Quick Test Run (Faster, Lower Resources)**

For rapid iteration and testing:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_quixer_training.sh \
  --qubits 4 \
  --layers 2 \
  --ansatz_layers 2 \
  --embedding_dim 64 \
  --max_seq_len 32 \
  --batch_size 16 \
  --epochs 5 \
  --lr 0.002 \
  --weight_decay 0.0001 \
  --dropout 0.10 \
  --scheduler cosine \
  --device cuda \
  --seed 42
```

---

## **Memory-Constrained Configuration**

If you encounter CUDA OOM errors:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_quixer_training.sh \
  --qubits 5 \
  --layers 2 \
  --ansatz_layers 3 \
  --embedding_dim 256 \
  --max_seq_len 32 \
  --batch_size 8 \
  --epochs 30 \
  --lr 0.002 \
  --weight_decay 0.0001 \
  --dropout 0.10 \
  --scheduler cosine \
  --device cuda \
  --seed 42
```

---

## **Hyperparameter Tuning Guide**

### 1. **Batch Size** (`--batch_size`)
- **Original:** 32 (effective 1024 with 32 contexts)
- **Genomics:** Start with 32, reduce to 16 or 8 if memory issues
- **Impact:** Larger = more stable gradients, but higher memory
- **GPU Memory Usage:** Exponential with qubits (2^qubits states to simulate)

### 2. **Number of Qubits** (`--qubits`)
- **Original:** 6 qubits
- **Genomics:** 6 is optimal, but can reduce to 4-5 for speed
- **Impact:** 
  - More qubits = richer quantum feature space
  - Memory scales as O(2^qubits) per batch element
  - 6 qubits → 64-dimensional quantum state
  - 4 qubits → 16-dimensional quantum state

### 3. **QSVT Layers** (`--layers`)
- **Original:** 3 (cubic polynomial)
- **Genomics:** 2-3 recommended
- **Impact:** 
  - Higher degree = more expressive nonlinearity
  - But increases training time linearly
  - Degree 3 captures up to 3-gram interactions

### 4. **Ansatz Layers** (`--ansatz_layers`)
- **Original:** 4 layers of "circuit 14"
- **Genomics:** 3-4 layers
- **Impact:** 
  - More layers = more expressible token embeddings
  - Each layer adds 4×qubits parameters per token
  - 4 layers × 6 qubits = 96 parameters per token

### 5. **Embedding Dimension** (`--embedding_dim`)
- **Original:** 512 for Quixer, 96/128 for baselines
- **Genomics:** 
  - 512 for full Quixer performance
  - 64-256 for faster training/memory savings
- **Impact:** Maps k-mer tokens to continuous embeddings before quantum encoding

### 6. **Sequence Length** (`--max_seq_len`)
- **Original:** 32 tokens
- **Genomics:** 32-64 tokens
- **Impact:** 
  - Longer sequences = more context for classification
  - But increases memory (LCU requires log2(n_tokens) control qubits)
  - Your k-mer sequences have variable length, padded/truncated to this

### 7. **Learning Rate** (`--lr`)
- **Original:** 0.002 for Quixer, 0.001 for Transformer
- **Genomics:** Start with 0.002
- **Tuning:** If loss plateaus, reduce by 2-5×

### 8. **Dropout** (`--dropout`)
- **Original:** 0.10 for Quixer/FNet, 0.30 for LSTM
- **Genomics:** 0.10-0.15
- **Impact:** Higher dropout prevents overfitting on small datasets

### 9. **Weight Decay** (`--weight_decay`)
- **Original:** 0.0001
- **Genomics:** 0.0001-0.001
- **Impact:** L2 regularization, helps with generalization

### 10. **Epochs** (`--epochs`)
- **Original:** 30 epochs (3h 45min on A100 for PTB)
- **Genomics:** 20-50 epochs
- **Monitoring:** Use validation accuracy to detect overfitting

---

## **Expected Performance**

### From the Paper (PTB Language Modeling):
- **Quixer (6 qubits, dim=512):** ~117 perplexity
- **FNet (dim=128):** ~115 perplexity
- **Transformer (dim=96):** ~108 perplexity
- **Training time:** ~3h 45min for 30 epochs on A100 GPU

### For Genomics Classification:
- **Target:** >80% test accuracy, F1-score >0.80
- **Baseline:** Random guessing = 50% (balanced binary classes)
- **Success criteria:** Quixer matches or exceeds classical baselines within ±3%
- **Training time estimate:** 1-3 hours per 30 epochs on L40S GPU

---

## **Training Tips**

1. **Start with a quick test run** (5 epochs, reduced params) to verify the pipeline works
2. **Monitor validation metrics** every epoch to detect overfitting
3. **Use gradient clipping** (automatically set to 1.0 in training script)
4. **Cosine annealing scheduler** warms up learning rate and reduces toward end
5. **Save best model** based on validation accuracy (automatic in script)
6. **Log postselection probabilities** if using full quantum simulation (not needed for classical simulation)

---

## **Hardware Requirements**

### Minimum:
- **GPU:** 8GB VRAM (e.g., NVIDIA RTX 3060)
- **RAM:** 16GB
- **Disk:** 5GB for dependencies + data

### Recommended:
- **GPU:** 24GB+ VRAM (L40S, A100, RTX 3090/4090)
- **RAM:** 32GB+
- **Disk:** 50GB

### Memory Scaling:
- 4 qubits, batch=16: ~4GB VRAM
- 6 qubits, batch=32: ~12GB VRAM
- 6 qubits, batch=64: ~24GB VRAM

---

## **Troubleshooting**

### CUDA Out of Memory
1. Reduce `--batch_size` (32 → 16 → 8)
2. Reduce `--qubits` (6 → 5 → 4)
3. Reduce `--embedding_dim` (512 → 256 → 128)
4. Reduce `--max_seq_len` (64 → 32)

### Training Too Slow
1. Reduce `--layers` (3 → 2)
2. Reduce `--ansatz_layers` (4 → 3 → 2)
3. Use smaller test configuration first

### Poor Accuracy
1. Increase `--epochs` (30 → 50)
2. Increase `--embedding_dim` (64 → 128 → 256)
3. Tune `--lr` (try 0.001 or 0.005)
4. Adjust `--dropout` (0.05-0.20 range)

### Overfitting (Val accuracy << Train accuracy)
1. Increase `--dropout` (0.10 → 0.15 → 0.20)
2. Increase `--weight_decay` (0.0001 → 0.001)
3. Reduce model complexity (`--qubits`, `--layers`)

---

## **References**

- Quixer Paper: https://arxiv.org/abs/2406.04305
- GitHub Repo: https://github.com/CQCL/Quixer
- TorchQuantum: https://github.com/mit-han-lab/torchquantum
