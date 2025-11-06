#!/usr/bin/env python3
"""
Prepare Classical Benchmark Data for Next-Token Prediction
===========================================================
Creates datasets for LSTM, Transformer, and other classical baselines
using the same RNA sequences as Quixer.

Task: Next-token prediction (autoregressive language modeling)
"""

import argparse
from pathlib import Path
from Bio import SeqIO
import numpy as np
import torch
from tqdm import tqdm
import json


def extract_kmers(sequence: str, k: int = 6) -> list:
    """Extract overlapping k-mers from sequence."""
    sequence = sequence.upper().replace('U', 'T')
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if all(n in 'ACGT' for n in kmer):
            kmers.append(kmer)
    return kmers


def kmer_to_token_id(kmer: str, vocab: dict) -> int:
    """Convert k-mer to token ID using vocabulary."""
    return vocab.get(kmer, vocab.get('<UNK>', 0))


def prepare_classical_data(
    fasta_path: str,
    output_dir: str,
    k: int = 6,
    window_size: int = 32,
    min_sequence_length: int = 200,
    max_sequences: int = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
):
    """
    Prepare data for classical baselines (LSTM, Transformer, etc.)
    
    Output format: PyTorch tensors with token sequences
    - sequences: [N, window_size] - token IDs
    - labels: [N] - next token ID (for next-token prediction)
    
    Args:
        fasta_path: Path to FASTA file with RNA sequences
        output_dir: Output directory for train/val/test splits
        k: k-mer size
        window_size: Number of k-mers per window
        min_sequence_length: Minimum sequence length (bp)
        max_sequences: Maximum sequences to process
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("CLASSICAL BENCHMARK DATA PREPARATION")
    print("="*70)
    print(f"Input FASTA: {fasta_path}")
    print(f"Output directory: {output_dir}")
    print(f"k-mer size: {k}")
    print(f"Window size: {window_size} k-mers")
    print()
    
    # Load sequences
    print("Loading sequences...")
    sequences = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_str = str(record.seq).upper().replace('U', 'T')
        if len(seq_str) >= min_sequence_length:
            sequences.append(seq_str)
            if max_sequences and len(sequences) >= max_sequences:
                break
    
    print(f"Loaded {len(sequences)} sequences")
    
    # Build vocabulary from all k-mers
    print("\nBuilding vocabulary...")
    vocab = {}
    vocab_id = 0
    
    for seq in tqdm(sequences, desc="Extracting k-mers"):
        kmers = extract_kmers(seq, k)
        for kmer in kmers:
            if kmer not in vocab:
                vocab[kmer] = vocab_id
                vocab_id += 1
    
    # Add special tokens
    vocab['<UNK>'] = vocab_id
    vocab_id += 1
    
    print(f"Vocabulary size: {len(vocab)}")
    
    # Extract all windows with token IDs
    print("\nExtracting k-mer windows...")
    all_sequences = []
    all_labels = []
    
    for seq in tqdm(sequences, desc="Processing sequences"):
        kmers = extract_kmers(seq, k)
        
        # Sliding window over k-mers
        for i in range(len(kmers) - window_size + 1):
            window = kmers[i:i+window_size]
            if len(window) == window_size:
                # Convert to token IDs
                token_ids = [kmer_to_token_id(kmer, vocab) for kmer in window]
                
                # Input: first window_size-1 tokens
                # Label: last token (what to predict)
                all_sequences.append(token_ids)
                all_labels.append(token_ids[-1])
    
    print(f"Extracted {len(all_sequences)} windows")
    
    # Convert to tensors
    sequences_tensor = torch.tensor(all_sequences, dtype=torch.long)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    
    # Shuffle and split
    print("\nShuffling and splitting data...")
    indices = np.random.permutation(len(sequences_tensor))
    
    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    # Save each split
    for split_name, split_indices in splits.items():
        split_sequences = sequences_tensor[split_indices]
        split_labels = labels_tensor[split_indices]
        
        # Save as PyTorch tensors
        output_file = output_dir / f"{split_name}.pt"
        torch.save({
            'sequences': split_sequences,
            'labels': split_labels,
            'task': 'next_token_prediction'
        }, output_file)
        
        print(f"\n{split_name.capitalize()} split:")
        print(f"  Samples: {len(split_sequences)}")
        print(f"  File: {output_file}")
    
    # Save vocabulary
    vocab_file = output_dir / 'vocab.json'
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"\nVocabulary saved: {vocab_file}")
    
    # Save metadata
    metadata = {
        'k': k,
        'window_size': window_size,
        'min_sequence_length': min_sequence_length,
        'n_sequences': len(sequences),
        'n_windows': len(all_sequences),
        'train_samples': len(train_indices),
        'val_samples': len(val_indices),
        'test_samples': len(test_indices),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': 1.0 - train_ratio - val_ratio,
        'vocabulary_size': len(vocab),
        'seed': seed,
        'task': 'next_token_prediction'
    }
    
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved: {metadata_file}")
    
    # Print example
    print("\nExample training window:")
    example_idx = train_indices[0]
    example_seq = sequences_tensor[example_idx].tolist()
    example_label = labels_tensor[example_idx].item()
    
    # Find k-mers for this example
    example_kmers = [k for k, v in vocab.items() if v in example_seq[:window_size-1]][:window_size-1]
    label_kmer = [k for k, v in vocab.items() if v == example_label][0]
    
    print(f"  Input tokens: {example_seq[:-1]}")
    print(f"  Target token: {example_label} (k-mer: {label_kmer})")
    
    print("\n" + "="*70)
    print("CLASSICAL BENCHMARK DATA READY")
    print("="*70)
    print(f"\nNext steps:")
    print(f"1. Train LSTM baseline:")
    print(f"   python train_classical_baselines.py \\")
    print(f"     --data_dir {output_dir} \\")
    print(f"     --model lstm")
    print(f"\n2. Train Transformer baseline:")
    print(f"   python train_classical_baselines.py \\")
    print(f"     --data_dir {output_dir} \\")
    print(f"     --model transformer")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare classical benchmark data for next-token prediction"
    )
    parser.add_argument(
        "--fasta",
        type=str,
        default="GRCh38_genomic_dataset/rna_sequences.fasta",
        help="Input FASTA file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="classical_benchmarks_data",
        help="Output directory"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="k-mer size"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=32,
        help="Number of k-mers per window"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=200,
        help="Minimum sequence length (bp)"
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Maximum sequences to process"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    prepare_classical_data(
        fasta_path=args.fasta,
        output_dir=args.output_dir,
        k=args.k,
        window_size=args.window_size,
        min_sequence_length=args.min_length,
        max_sequences=args.max_sequences,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
