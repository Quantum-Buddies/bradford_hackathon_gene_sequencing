#!/usr/bin/env python3
"""
Quantize Lambeq Embeddings to Token IDs (Hybrid Approach)
==========================================================
Converts continuous lambeq quantum embeddings into discrete tokens via clustering.

This hybrid approach combines:
1. Lambeq's quantum compositional embeddings (captures k-mer structure)
2. Vector quantization via k-means clustering (creates discrete tokens)
3. Quixer's quantum attention (processes the quantized tokens)

The result: Quantum features + Quantum transformer = Best of both worlds!
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import pickle


def load_lambeq_embeddings(embeddings_dir: Path, split: str):
    """Load lambeq per-k-mer embeddings for a split.
    
    Returns:
        embeddings: [N, max_kmers, embedding_dim] numpy array
        labels: [N] numpy array
    """
    file_path = embeddings_dir / f"{split}.pt"
    data = torch.load(file_path)
    embeddings = data['embeddings'].numpy()
    labels = data['labels'].numpy()
    
    # Check if per-k-mer format
    if len(embeddings.shape) == 3:
        print(f"  Per-k-mer embeddings detected: {embeddings.shape}")
    else:
        print(f"  WARNING: Old sentence-level format detected: {embeddings.shape}")
        print(f"  This will result in repeated tokens - consider re-encoding with per-k-mer mode")
    
    return embeddings, labels


def quantize_embeddings(
    train_embeddings: np.ndarray,
    val_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    n_clusters: int = 512,
    random_state: int = 42,
    batch_size: int = 1000
):
    """
    Quantize continuous embeddings to discrete token IDs using k-means clustering.
    
    Handles both formats:
    - Per-k-mer: [N, seq_len, D] → cluster each position independently
    - Sentence-level: [N, D] → cluster whole sentences (legacy)
    
    Args:
        train_embeddings: Training embeddings [N, D] or [N, seq_len, D]
        val_embeddings: Validation embeddings [M, D] or [M, seq_len, D]
        test_embeddings: Test embeddings [K, D] or [K, seq_len, D]
        n_clusters: Number of clusters (vocabulary size for Quixer)
        random_state: Random seed
        batch_size: Batch size for MiniBatchKMeans
    
    Returns:
        train_tokens, val_tokens, test_tokens, kmeans_model, cluster_centers
    """
    print(f"\nQuantizing embeddings with k-means clustering...")
    print(f"  Clusters (vocabulary size): {n_clusters}")
    
    # Check format: per-k-mer [N, seq_len, D] or sentence-level [N, D]
    per_kmer_mode = len(train_embeddings.shape) == 3
    
    if per_kmer_mode:
        n_samples, seq_len, emb_dim = train_embeddings.shape
        print(f"  Per-k-mer mode: {n_samples} samples × {seq_len} k-mers × {emb_dim} dims")
        
        # Flatten to [N * seq_len, emb_dim] for clustering
        train_flat = train_embeddings.reshape(-1, emb_dim)
        val_flat = val_embeddings.reshape(-1, emb_dim)
        test_flat = test_embeddings.reshape(-1, emb_dim)
        
        print(f"  Flattened training data: {train_flat.shape}")
    else:
        print(f"  Sentence-level mode (legacy): {train_embeddings.shape}")
        train_flat = train_embeddings
        val_flat = val_embeddings
        test_flat = test_embeddings
    
    # Fit k-means on flattened training data
    print("\nFitting k-means on training embeddings...")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        max_iter=100,
        verbose=1,
        n_init=3
    )
    kmeans.fit(train_flat)
    
    # Get cluster centers (this becomes our "quantum codebook")
    cluster_centers = kmeans.cluster_centers_
    print(f"\nCluster centers shape: {cluster_centers.shape}")
    
    # Assign each embedding to nearest cluster (quantization)
    print("\nQuantizing embeddings to token IDs...")
    train_tokens = kmeans.predict(train_flat)
    val_tokens = kmeans.predict(val_flat)
    test_tokens = kmeans.predict(test_flat)
    
    # Reshape back to sequences if per-k-mer mode
    if per_kmer_mode:
        train_tokens = train_tokens.reshape(n_samples, seq_len)
        val_tokens = val_tokens.reshape(val_embeddings.shape[0], val_embeddings.shape[1])
        test_tokens = test_tokens.reshape(test_embeddings.shape[0], test_embeddings.shape[1])
        print(f"  Reshaped to sequences: train={train_tokens.shape}, val={val_tokens.shape}, test={test_tokens.shape}")
    
    # Compute quantization metrics
    if per_kmer_mode:
        # Compute error for flattened data
        train_distances = np.linalg.norm(
            train_flat - cluster_centers[train_tokens.flatten()], axis=1
        )
    else:
        train_distances = np.linalg.norm(
            train_embeddings - cluster_centers[train_tokens], axis=1
        )
    avg_quantization_error = np.mean(train_distances)
    
    print(f"\nQuantization complete!")
    print(f"  Token range: [{train_tokens.min()}, {train_tokens.max()}]")
    print(f"  Unique tokens (train): {len(np.unique(train_tokens.flatten()))}")
    print(f"  Average quantization error: {avg_quantization_error:.4f}")
    
    # Check cluster utilization
    token_counts = np.bincount(train_tokens.flatten(), minlength=n_clusters)
    unused_clusters = np.sum(token_counts == 0)
    print(f"  Unused clusters: {unused_clusters}/{n_clusters}")
    
    if per_kmer_mode:
        # Check token diversity per sample
        unique_per_sample = [len(np.unique(seq)) for seq in train_tokens]
        print(f"  Avg unique tokens per sequence: {np.mean(unique_per_sample):.1f}")
    
    return train_tokens, val_tokens, test_tokens, kmeans, cluster_centers


def create_token_sequences(
    tokens: np.ndarray,
    labels: np.ndarray,
    target_seq_len: int = 32,
    pad_token: int = 0
):
    """
    Prepare token sequences for Quixer.
    
    If tokens are already sequences [N, seq_len], truncate or pad to target length.
    If tokens are flat [N], repeat to create sequence (legacy mode).
    
    Args:
        tokens: Token IDs [N] or [N, seq_len]
        labels: Class labels [N]
        target_seq_len: Target sequence length for Quixer
        pad_token: Token ID for padding
    
    Returns:
        sequences: [N, target_seq_len] of token IDs
        labels: [N] unchanged
    """
    if len(tokens.shape) == 2:
        # Already sequences - truncate or pad
        n_samples, current_seq_len = tokens.shape
        
        if current_seq_len > target_seq_len:
            # Truncate
            sequences = tokens[:, :target_seq_len]
        elif current_seq_len < target_seq_len:
            # Pad
            pad_width = ((0, 0), (0, target_seq_len - current_seq_len))
            sequences = np.pad(tokens, pad_width, mode='constant', constant_values=pad_token)
        else:
            sequences = tokens
    else:
        # Flat tokens - repeat for backward compatibility
        n_samples = len(tokens)
        sequences = np.tile(tokens.reshape(-1, 1), (1, target_seq_len))
        print(f"  WARNING: Repeating single token per sample (legacy mode)")
    
    return sequences, labels


def save_quantized_data(
    output_dir: Path,
    train_seqs: np.ndarray,
    train_labels: np.ndarray,
    val_seqs: np.ndarray,
    val_labels: np.ndarray,
    test_seqs: np.ndarray,
    test_labels: np.ndarray,
    kmeans_model,
    cluster_centers: np.ndarray,
    metadata: dict
):
    """Save quantized data and model."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save sequences as PyTorch tensors
    print(f"\nSaving quantized data to {output_dir}...")
    
    torch.save({
        'sequences': torch.from_numpy(train_seqs).long(),
        'labels': torch.from_numpy(train_labels).long()
    }, output_dir / 'train.pt')
    
    torch.save({
        'sequences': torch.from_numpy(val_seqs).long(),
        'labels': torch.from_numpy(val_labels).long()
    }, output_dir / 'val.pt')
    
    torch.save({
        'sequences': torch.from_numpy(test_seqs).long(),
        'labels': torch.from_numpy(test_labels).long()
    }, output_dir / 'test.pt')
    
    # Save k-means model and cluster centroids for downstream use
    with open(output_dir / 'kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans_model, f)

    torch.save(
        {
            'centroids': torch.from_numpy(cluster_centers.astype(np.float32))
        },
        output_dir / 'cluster_centers.pt'
    )
    
    # Save metadata
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Saved train.pt: {train_seqs.shape}")
    print(f"  ✓ Saved val.pt: {val_seqs.shape}")
    print(f"  ✓ Saved test.pt: {test_seqs.shape}")
    print(f"  ✓ Saved kmeans_model.pkl")
    print(f"  ✓ Saved metadata.json")
    print(f"  ✓ Saved cluster_centers.pt: {cluster_centers.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize lambeq embeddings to token IDs for Quixer hybrid approach"
    )
    
    parser.add_argument(
        '--embeddings_dir',
        type=str,
        default='/scratch/cbjp404/bradford_hackathon_2025/lambeq_embeddings',
        help='Directory containing lambeq embeddings'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/scratch/cbjp404/bradford_hackathon_2025/quantized_embeddings',
        help='Output directory for quantized tokens'
    )
    parser.add_argument(
        '--n_clusters',
        type=int,
        default=512,
        help='Number of clusters (vocabulary size for Quixer)'
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=32,
        help='Sequence length for Quixer (tokens will be repeated to this length)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Batch size for MiniBatchKMeans'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("QUANTIZE LAMBEQ EMBEDDINGS TO TOKEN IDS (HYBRID APPROACH)")
    print("="*70)
    print(f"Input: {args.embeddings_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Clusters (vocab size): {args.n_clusters}")
    print(f"Sequence length: {args.seq_len}")
    print()
    
    # Load lambeq embeddings
    print("Loading lambeq embeddings...")
    embeddings_dir = Path(args.embeddings_dir)
    
    train_emb, train_labels = load_lambeq_embeddings(embeddings_dir, 'train')
    val_emb, val_labels = load_lambeq_embeddings(embeddings_dir, 'val')
    test_emb, test_labels = load_lambeq_embeddings(embeddings_dir, 'test')
    
    print(f"  Train: {train_emb.shape}")
    print(f"  Val: {val_emb.shape}")
    print(f"  Test: {test_emb.shape}")
    
    # Quantize embeddings
    train_tokens, val_tokens, test_tokens, kmeans, cluster_centers = quantize_embeddings(
        train_emb, val_emb, test_emb,
        n_clusters=args.n_clusters,
        random_state=args.seed,
        batch_size=args.batch_size
    )
    
    # Create sequences
    print(f"\nCreating token sequences (target length={args.seq_len})...")
    train_seqs, train_labels = create_token_sequences(
        train_tokens, train_labels, target_seq_len=args.seq_len
    )
    val_seqs, val_labels = create_token_sequences(
        val_tokens, val_labels, target_seq_len=args.seq_len
    )
    test_seqs, test_labels = create_token_sequences(
        test_tokens, test_labels, target_seq_len=args.seq_len
    )
    
    print(f"  Train sequences: {train_seqs.shape}")
    print(f"  Val sequences: {val_seqs.shape}")
    print(f"  Test sequences: {test_seqs.shape}")
    
    # Save
    # Determine embedding format
    per_kmer_mode = len(train_emb.shape) == 3
    emb_dim = train_emb.shape[2] if per_kmer_mode else train_emb.shape[1]
    
    metadata = {
        'n_clusters': args.n_clusters,
        'seq_len': args.seq_len,
        'embedding_dim': emb_dim,
        'vocabulary_size': args.n_clusters,
        'n_classes': len(np.unique(train_labels)),
        'source': str(args.embeddings_dir),
        'quantization_method': 'MiniBatchKMeans',
        'seed': args.seed,
        'per_kmer_mode': per_kmer_mode,
        'original_seq_len': train_emb.shape[1] if per_kmer_mode else None
    }
    
    save_quantized_data(
        Path(args.output_dir),
        train_seqs, train_labels,
        val_seqs, val_labels,
        test_seqs, test_labels,
        kmeans,
        cluster_centers,
        metadata
    )
    
    print("\n" + "="*70)
    print("QUANTIZATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print(f"1. Train Quixer on quantized tokens:")
    print(f"   python train_quixer_hybrid.py \\")
    print(f"     --data_dir {args.output_dir} \\")
    print(f"     --vocab_size {args.n_clusters} \\")
    print(f"     --seq_len {args.seq_len}")
    print("\n2. This combines:")
    print("   ✓ Lambeq's quantum compositional embeddings")
    print("   ✓ K-means quantization (discrete tokens)")
    print("   ✓ Quixer's quantum attention mechanism")
    print("="*70)


if __name__ == "__main__":
    main()
