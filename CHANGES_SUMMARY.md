# Changes Summary for Next-Token Prediction

## ✅ Completed Changes

### **1. prepare_autoregressive_data.py**
- ✅ Fixed to create subdirectories (`autoregressive_data/train/`, etc.)
- ✅ Generates dummy `labels.txt` files (all 0s, required by lambeq)
- ✅ Saves sentences in correct format for lambeq

### **2. train_quixer_hybrid.py**
- ✅ Added `--task` argument (`classification` or `autoregressive`)
- ✅ Modified `train_epoch()` to handle both tasks
- ✅ Modified `evaluate()` to handle both tasks  
- ✅ Added task-specific data handling:
  - Classification: `inputs = sequences`, `targets = labels`
  - Autoregressive: `inputs = sequences[:, :-1]`, `targets = sequences[:, -1]`

### **3. Files That DON'T Need Changes**
- ✅ `lambeq_encoder.py` – Works as-is
- ✅ `quantize_lambeq_embeddings.py` – Works as-is
- ✅ `quixer_wrapper.py` – Output dimension (512) is correct!

---

## ⚠️ Still TODO

### **tune_quixer_hybrid.py**
Same changes needed as `train_quixer_hybrid.py`:

1. Add `--task` argument to parser
2. Modify `train_epoch()` function signature
3. Modify `evaluate()` function signature  
4. Update all calls to `train_epoch()` and `evaluate()` to pass `task=args.task`

This is straightforward - just copy the same changes from `train_quixer_hybrid.py`.

---

## ❌ Files You Don't Need

### **hybrid_quixer_classifier.py**
**Status**: Not needed

**Reason**: You're using the real Torch Quantum Quixer via `quixer_wrapper.py`, not this pure PyTorch fallback.

**Action**: Can delete or ignore

### **train_quixer_genomics.py**
**Status**: Old/different script

**Reason**: You have `train_quixer_hybrid.py` which is the correct one

**Action**: Check if it's outdated, may be safe to ignore

---

## Complete Pipeline

### **Quick Test (2-3 hours)**

```bash
# 1. Prepare autoregressive data
python prepare_autoregressive_data.py \
  --fasta GRCh38_genomic_dataset/rna_sequences.fasta \
  --output_dir autoregressive_data

# 2. Encode with lambeq (qrisp-jax environment)
module load miniforge
conda activate qrisp-jax

python lambeq_encoder.py \
  --data_dir autoregressive_data \
  --output_dir lambeq_embeddings_autoregressive \
  --embedding_dim 64 \
  --layers 2

# 3. Quantize (quixer environment)
conda activate quixer

python quantize_lambeq_embeddings.py \
  --embeddings_dir lambeq_embeddings_autoregressive \
  --output_dir quantized_embeddings_autoregressive \
  --n_clusters 512 \
  --seq_len 32

# 4. Train Quixer
python train_quixer_hybrid.py \
  --data_dir quantized_embeddings_autoregressive \
  --task autoregressive \
  --epochs 50 \
  --init_strategy small_gaussian \
  --use_layerwise_training
```

### **Expected Output**

```
Task: autoregressive
Init strategy: small_gaussian
Layerwise training: True

Epoch 1/50:
  Train loss: 6.2314, acc: 0.4%   # Low accuracy is NORMAL for 512-way
  Val loss: 6.1894, acc: 0.5%

Epoch 10/50:
  Train loss: 4.8234, acc: 5.2%   # Improving!
  Val loss: 4.7891, acc: 5.4%

Epoch 50/50:
  Train loss: 3.2134, acc: 18.3%  # Good for 512-way!
  Val loss: 3.3245, acc: 17.1%
  Perplexity: 27.7               # Much better than random (512)
```

---

## Comparison: Classification vs Autoregressive

| Aspect | Binary Classification | Next-Token Prediction |
|--------|----------------------|----------------------|
| **Task** | Promoter vs non-promoter | Predict next k-mer |
| **Input** | Full sequence (32 tokens) | First 31 tokens |
| **Target** | Separate labels (0 or 1) | 32nd token (0-511) |
| **Output** | 2 logits | 512 logits |
| **Good accuracy** | 75-85% | 15-25% (512-way!) |
| **Better metric** | F1-score | Perplexity |
| **Random baseline** | 50% | 0.2% (1/512) |

---

## Troubleshooting

### **Issue: "labels.txt not found"**
**Solution**: Run `prepare_autoregressive_data.py` again with the fixed version

### **Issue: "Shape mismatch in evaluate()"**
**Solution**: Make sure `--task autoregressive` is passed

### **Issue: "Accuracy only 0.2%"**
**Check**: 
1. Is `--task autoregressive` set?
2. Are inputs/targets correctly split in training loop?
3. Is model actually training (loss decreasing)?

### **Issue: "Still getting 56% accuracy"**
**Likely**: You're running classification task, not autoregressive
**Fix**: Add `--task autoregressive` flag

---

## Next Steps

1. ✅ **Data ready**: You have 10k RNA sequences downloaded
2. ⏳ **Run pipeline**: Execute the 4 commands above
3. ⏳ **Fix tune script**: Apply same changes to `tune_quixer_hybrid.py` (optional, can use train script first)
4. ⏳ **Evaluate**: Compare perplexity, top-k accuracy

Good luck! 🚀
