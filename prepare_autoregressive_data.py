#!/usr/bin/env python3
"""
Prepare Real Genomic Sequences for Next-Token Prediction
==========================================================
Converts RNA FASTA sequences into autoregressive training data.

Task: Given k-mers 1 to n-1, predict k-mer n
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


def sequence_to_sentence(sequence: str, k: int = 6) -> str:
    """Convert sequence to space-separated k-mer sentence."""
    kmers = extract_kmers(sequence, k)
    return ' '.join(kmers)


def prepare_autoregressive_windows(
    fasta_path: str,
    output_dir: str,
    k: int = 6,
    window_size: int = 32,  # Number of k-mers per window
    min_sequence_length: int = 200,  # Min bp (33+ k-mers)
    max_sequences: int = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
):
    """
    Prepare autoregressive training data from FASTA sequences.
    
    For each window of k-mers:
    - Input: k-mers 0 to window_size-2 (31 tokens)
    - Target: k-mer window_size-1 (32nd token to predict)
    
    Args:
        fasta_path: Path to FASTA file with RNA sequences
        output_dir: Output directory for train/val/test splits
        k: k-mer size
        window_size: Number of k-mers per training window
        min_sequence_length: Minimum sequence length in bp
        max_sequences: Maximum sequences to process (None = all)
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        seed: Random seed
    """
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("AUTOREGRESSIVE DATA PREPARATION")
    print("="*70)
    print(f"Input FASTA: {fasta_path}")
    print(f"Output directory: {output_dir}")
    print(f"k-mer size: {k}")
    print(f"Window size: {window_size} k-mers")
    print(f"Min sequence length: {min_sequence_length} bp")
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
    
    # Extract all windows
    print("\nExtracting k-mer windows...")
    all_sentences = []
    
    for seq in tqdm(sequences, desc="Processing sequences"):
        kmers = extract_kmers(seq, k)
        
        # Sliding window over k-mers
        for i in range(len(kmers) - window_size + 1):
            window = kmers[i:i+window_size]
            if len(window) == window_size:
                # Create sentence for lambeq encoding
                sentence = ' '.join(window)
                all_sentences.append(sentence)
    
    print(f"Extracted {len(all_sentences)} windows")
    
    # Shuffle and split
    print("\nShuffling and splitting data...")
    indices = np.random.permutation(len(all_sentences))
    
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
    
    # Save each split (create subdirectories as expected by lambeq)
    for split_name, split_indices in splits.items():
        sentences = [all_sentences[i] for i in split_indices]
        
        # Create split subdirectory
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Save sentences for lambeq encoding
        sentences_file = split_dir / "sentences.txt"
        with open(sentences_file, 'w') as f:
            for sentence in sentences:
                f.write(sentence + '\n')
        
        # Save dummy labels (all 0s) - not used for next-token prediction
        # but required by lambeq_encoder.py
        labels_file = split_dir / "labels.txt"
        with open(labels_file, 'w') as f:
            for _ in sentences:
                f.write('0\n')
        
        print(f"\n{split_name.capitalize()} split:")
        print(f"  Samples: {len(sentences)}")
        print(f"  Directory: {split_dir}")
        print(f"  Files: sentences.txt, labels.txt")
    
    # Save metadata
    metadata = {
        'k': k,
        'window_size': window_size,
        'min_sequence_length': min_sequence_length,
        'n_sequences': len(sequences),
        'n_windows': len(all_sentences),
        'train_samples': len(train_indices),
        'val_samples': len(val_indices),
        'test_samples': len(test_indices),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': 1.0 - train_ratio - val_ratio,
        'seed': seed,
        'task': 'next_kmer_prediction'
    }
    
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to {metadata_file}")
    
    # Print example
    print("\nExample training window:")
    example = all_sentences[train_indices[0]]
    kmers = example.split()
    print(f"  Input (first {window_size-1} k-mers): {' '.join(kmers[:-1])}")
    print(f"  Target (k-mer to predict): {kmers[-1]}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print(f"1. Encode with lambeq:")
    print(f"   python lambeq_encoder.py --data_dir {output_dir}")
    print(f"\n2. Quantize embeddings:")
    print(f"   python quantize_lambeq_embeddings.py \\")
    print(f"     --embeddings_dir lambeq_embeddings \\")
    print(f"     --output_dir quantized_embeddings_autoregressive")
    print(f"\n3. Train Quixer for next-token prediction:")
    print(f"   python tune_quixer_hybrid.py \\")
    print(f"     --data_dir quantized_embeddings_autoregressive")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Prepare autoregressive training data from genomic sequences"
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
        default="autoregressive_data",
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
    
    prepare_autoregressive_windows(
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
