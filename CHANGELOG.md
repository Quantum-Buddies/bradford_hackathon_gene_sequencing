# Changelog: Quixer Hybrid Accuracy Fix

## Version 2.0: Per-K-mer Embeddings + Centroid Initialization (Current)

### Major Changes

#### 1. Per-K-mer Embedding Generation (`lambeq_encoder.py`)
**Problem**: Original pipeline generated one embedding per entire sequence, collapsing all compositional structure.
- Input: k-mer sentence "AAAAAA CCCCCC GGGGGG ..."
- Old output: Single 512-dim vector (all positional info lost)
- New output: Sequence of 64-dim vectors, one per k-mer

**Solution**: Modified `GenomicLambeqEncoder` to:
- Add `_evaluate_kmer_with_model()` for single k-mer encoding
- Add `encode_kmer_sequence()` to process sequences of k-mers
- Output shape: `[N_samples, max_kmers, 64]` instead of `[N_samples, 512]`
- Preserve positional structure for Quixer's quantum attention

**Impact**:
- ✅ Quixer now receives diverse token representations
- ✅ Quantum attention can operate on meaningful sequence structure
- ✅ Expected accuracy improvement from ~50% (random) to ≥80%

#### 2. Vector Quantization with Centroid Initialization (`quantize_lambeq_embeddings.py`)
**Problem**: Quixer's embedding layer initialized with random weights, no connection to lambeq encodings.

**Solution**: Implemented quantization pipeline:
- Load per-k-mer embeddings `[N*max_kmers, 64]`
- Fit MiniBatchKMeans with 512 clusters
- Assign each embedding to nearest centroid
- Create token sequences `[N, seq_len]` with token IDs
- **Save cluster centroids as `cluster_centers.pt`**
- Validate cluster utilization and token diversity

**Key Features**:
- Flattens per-k-mer embeddings for clustering
- Reshapes token IDs back to sequences
- Handles variable-length sequences via truncation/padding
- Saves metadata including `per_kmer_mode` flag
- Exports centroids for embedding initialization

**Output Files**:
```
quantized_embeddings/
├── train.pt              # Token sequences + labels
├── val.pt
├── test.pt
├── cluster_centers.pt    # 512 × 64 centroid matrix
├── kmeans_model.pkl      # Fitted k-means model
└── metadata.json         # Configuration and metadata
```

#### 3. Embedding Layer Initialization (`train_quixer_hybrid.py`, `tune_quixer_hybrid.py`)
**Problem**: Embedding layer started with Xavier uniform initialization, random vectors.

**Solution**: Load and apply cluster centroids:
```python
# Load centroids
cluster_file = data_dir / 'cluster_centers.pt'
payload = torch.load(cluster_file)
cluster_centers = payload['centroids'].float()

# Initialize embedding layer
embedding_weight = model.quixer.embedding.weight
centroids = cluster_centers.to(embedding_weight.device)
with torch.no_grad():
    embedding_weight.copy_(centroids)
```

**Impact**:
- ✅ Token IDs now map to meaningful lambeq-derived vectors
- ✅ Linear layer WE learns better angle transformations
- ✅ Quantum circuits receive rich input features
- ✅ Training converges faster with better initialization

### Minor Changes

#### Fixed Hyperparameter Handling
- `tune_quixer_hybrid.py`: Fixed `embedding_dim` to use metadata value instead of sampling
- Both scripts: Validate centroid shape before copying to embedding layer
- Both scripts: Graceful fallback if centroids not found (warns user)

#### Improved Metadata Tracking
- `quantize_lambeq_embeddings.py`: Added `per_kmer_mode` flag to metadata
- All scripts: Consistent metadata loading and validation
- Better error messages for shape mismatches

### File Changes Summary

| File | Changes | Lines Modified |
|------|---------|-----------------|
| `lambeq_encoder.py` | Per-k-mer encoding, removed worker functions | 79-390 |
| `quantize_lambeq_embeddings.py` | Centroid saving, per-k-mer handling | 25-359 |
| `train_quixer_hybrid.py` | Centroid loading and initialization | 227-325 |
| `tune_quixer_hybrid.py` | Centroid loading, fixed embedding_dim | 43-250 |
| `README.md` | Updated architecture and pipeline docs | Multiple |
| `ARCHITECTURE.md` | New comprehensive architecture guide | New file |
| `.gitignore` | Git configuration | New file |

### Data Flow

**Old Pipeline** (v1.0):
```
k-mer sequences
    ↓
lambeq (sentence-level)
    ↓
[N, 512] embeddings (collapsed structure)
    ↓
Quantization (single token per sample)
    ↓
Quixer (all tokens identical) → ~50% accuracy
```

**New Pipeline** (v2.0):
```
k-mer sequences
    ↓
lambeq (per-k-mer)
    ↓
[N, max_kmers, 64] embeddings (preserved structure)
    ↓
Quantization (diverse tokens per sample)
    ↓
Centroid-initialized Quixer (meaningful embeddings) → ≥80% accuracy
```

### Testing & Validation

#### Quantization Output Validation
- Cluster utilization: All 512 clusters should be used
- Token diversity: Each sequence should have multiple unique tokens
- Quantization error: Should be reasonable (< 10% of embedding norm)
- Shape validation: Centroids must match embedding layer dimensions

#### Training Validation
- Embedding initialization: Check that centroids are copied correctly
- Loss curves: Should decrease smoothly (not random noise)
- Accuracy: Should improve beyond 50% baseline
- Convergence: Should reach plateau within 50 epochs

### Backward Compatibility

⚠️ **Breaking Changes**:
- Old `lambeq_embeddings/` format incompatible with new quantization
- Must re-run `lambeq_encoder.py` to generate per-k-mer embeddings
- Must re-run `quantize_lambeq_embeddings.py` to generate centroids

✅ **Migration Path**:
1. Delete old `lambeq_embeddings/` directory
2. Delete old `quantized_embeddings/` directory
3. Run: `python lambeq_encoder.py`
4. Run: `python quantize_lambeq_embeddings.py --n_clusters 512 --seq_len 32`
5. Run: `python train_quixer_hybrid.py` (will auto-load centroids)

### Performance Improvements

**Expected Results**:
| Metric | Old (v1.0) | New (v2.0) | Improvement |
|--------|-----------|-----------|------------|
| Test Accuracy | ~50% | ≥80% | +30% |
| F1-Score | ~0.50 | >0.80 | +0.30 |
| Convergence | Slow | Fast | 2-3× faster |
| Parameter Efficiency | N/A | <500K | Maintained |

### Documentation

**New Files**:
- `ARCHITECTURE.md`: Comprehensive pipeline architecture with diagrams
- `CHANGELOG.md`: This file
- `.gitignore`: Git configuration

**Updated Files**:
- `README.md`: Highlighted new per-k-mer and centroid features
- `PER_KMER_PIPELINE.md`: Existing documentation (still relevant)

### Known Issues & Limitations

1. **Memory Usage**: Per-k-mer embeddings increase memory footprint
   - Mitigation: Process in batches, reduce max_kmers if needed

2. **Quantization Error**: Some information lost during k-means clustering
   - Mitigation: Use 512 clusters (empirically determined)

3. **Centroid Initialization**: Assumes centroids are meaningful
   - Validation: Check cluster utilization and token diversity

### Future Work

1. **Adaptive Cluster Count**: Optimize n_clusters based on data
2. **Learned Quantization**: Replace k-means with learned quantization
3. **Hybrid Embeddings**: Combine quantized + continuous embeddings
4. **Multi-Scale Encoding**: Encode at multiple k-mer scales
5. **Attention Visualization**: Analyze what Quixer attends to

### References

- Quixer paper: arXiv:2406.04305
- lambeq documentation: https://docs.quantinuum.com/lambeq/
- MiniBatchKMeans: scikit-learn clustering

---

**Version**: 2.0  
**Date**: 2025-11-05  
**Status**: Ready for testing  
**Next Steps**: Run full training pipeline and validate accuracy improvements
