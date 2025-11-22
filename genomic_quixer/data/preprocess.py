"""
Build genomic dataset for next-base prediction using K-MER tokenization.

This script tokenizes DNA sequences into non-overlapping k-mers (default k=4)
and creates training samples for next-token prediction.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

import torch
from tqdm import tqdm
import numpy as np


# K-mer configuration
K_MER = 4
BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
VOCAB_SIZE = 4 ** K_MER


def load_fasta_sequences(fasta_path: Path) -> List[Tuple[str, str]]:
    """Load sequences from FASTA file."""
    sequences = []
    current_header = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append((current_header, ''.join(current_seq)))
                
                current_header = line[1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line.upper())
        
        if current_header is not None:
            sequences.append((current_header, ''.join(current_seq)))
    
    return sequences


def filter_sequences_atcg_only(sequences: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Filter out sequences containing non-ATCG characters."""
    print("\n1.5. Filtering for ATCG-only sequences...")
    filtered_sequences = []
    atcg_chars = set(BASE_TO_INT.keys())
    initial_count = len(sequences)

    for header, seq in tqdm(sequences, desc="Filtering sequences"):
        if all(char.upper() in atcg_chars for char in seq):
            filtered_sequences.append((header, seq))
    
    final_count = len(filtered_sequences)
    removed_count = initial_count - final_count
    print(f"  Removed {removed_count:,} sequences containing non-ATCG characters.")
    print(f"  Remaining sequences: {final_count:,}")
    
    return filtered_sequences


def encode_kmer(kmer: str) -> int:
    """Convert a k-mer string (e.g., 'ACGT') to an integer ID."""
    val = 0
    for char in kmer:
        val = val * 4 + BASE_TO_INT[char]
    return val


def tokenize_sequence_kmers(seq: str, k: int = K_MER) -> torch.Tensor:
    """Convert DNA sequence to non-overlapping k-mer token IDs.
    
    Args:
        seq: DNA sequence string
        k: k-mer size
    
    Returns:
        Tensor of token IDs [seq_length // k] (int16)
    """
    # Truncate sequence to be divisible by k
    n_tokens = len(seq) // k
    if n_tokens == 0:
        return torch.tensor([], dtype=torch.short)
        
    truncated_len = n_tokens * k
    seq = seq[:truncated_len]
    
    # Convert to k-mers
    tokens = []
    for i in range(0, truncated_len, k):
        kmer = seq[i:i+k]
        tokens.append(encode_kmer(kmer))
    
    return torch.tensor(tokens, dtype=torch.short)


def main():
    parser = argparse.ArgumentParser(description="Preprocess genomic data for Quixer")
    parser.add_argument('--input_fasta', type=str, default='/scratch/cbjp404/bradford_hackathon_gene_sequencing/GRCh38_genomic_dataset/rna_sequences.fasta', help='Path to input FASTA file')
    parser.add_argument('--output_dir', type=str, default='/scratch/cbjp404/bradford_hackathon_gene_sequencing/genomic_data_preprocessed_kmers', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--k_mer', type=int, default=K_MER, help='K-mer size')
    args = parser.parse_args()

    # Update global K_MER if argument provided (though encode_kmer relies on global BASE_TO_INT which is constant)
    # But let's respect the args for the tokenization function
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    global K_MER 
    global VOCAB_SIZE
    K_MER = args.k_mer
    VOCAB_SIZE = 4 ** K_MER

    print("=" * 70)
    print(f"Pre-processing Genomic Dataset: {K_MER}-mer Tokenization")
    print(f"Vocab Size: {VOCAB_SIZE:,}")
    print("=" * 70)
    
    # Load sequences
    print("\n1. Loading sequences...")
    sequences = load_fasta_sequences(Path(args.input_fasta))
    print(f"  Loaded {len(sequences):,} sequences")

    # Filter for ATCG-only sequences
    sequences = filter_sequences_atcg_only(sequences)

    # All sequences
    all_sequences = [seq for _, seq in sequences]

    # Tokenize and concatenate all sequences
    print(f"\n2. Tokenizing into non-overlapping {K_MER}-mers...")
    all_tokens = []
    seq_lengths = []
    
    total_bases_processed = 0
    for seq in tqdm(all_sequences, desc="Tokenizing"):
        tokens = tokenize_sequence_kmers(seq, k=K_MER)
        if len(tokens) > 0:
            all_tokens.append(tokens)
            seq_lengths.append(len(tokens))
            total_bases_processed += len(seq)
    
    if not all_tokens:
        print("Error: No valid tokens generated!")
        return

    concatenated_tokens = torch.cat(all_tokens)
    total_tokens = len(concatenated_tokens)
    
    print(f"  Total tokens generated: {total_tokens:,}")
    print(f"  Effective context window (128 tokens) = {128 * K_MER} bp")

    # Save data
    print(f"\n3. Saving pre-processed data to {output_dir}...")
    
    # Save as int16 to save space (vocab < 32k)
    torch.save(concatenated_tokens, output_dir / "all_tokens.pt")
    
    # Save metadata
    metadata = {
        'vocab_size': VOCAB_SIZE,
        'k_mer': K_MER,
        'n_sequences': len(seq_lengths),
        'total_tokens': total_tokens,
        'seq_lengths': seq_lengths,
        'seed': args.seed,
        'token_dtype': 'int16'
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("Dataset pre-processing complete!")
    print(f"Files saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
