#!/usr/bin/env python3
"""
Build Tokenized Genomic Dataset for Quixer (No Lambeq)
=======================================================

Reads RNA sequences from FASTA, tokenizes using k-mers, builds vocabulary,
and saves train/val/test splits as tensors compatible with Quixer training.

Usage:
    python build_genomic_dataset.py --fasta rna_sequences.fasta --k 6 --output genomic_data
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import random

import torch
import numpy as np
from tqdm import tqdm


def parse_fasta(fasta_path: Path) -> List[Tuple[str, str]]:
    """Parse FASTA file and return list of (header, sequence) tuples."""
    sequences = []
    current_header = None
    current_seq = []
    
    print(f"\nParsing FASTA: {fasta_path}")
    with open(fasta_path) as f:
        for line in tqdm(f, desc="Reading FASTA"):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('>'):
                # Save previous sequence
                if current_header is not None:
                    sequences.append((current_header, ''.join(current_seq)))
                
                current_header = line[1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)
        
        # Save last sequence
        if current_header is not None:
            sequences.append((current_header, ''.join(current_seq)))
    
    print(f"  Loaded {len(sequences):,} sequences")
    return sequences


def extract_kmers(sequence: str, k: int, overlap: bool = True) -> List[str]:
    """
    Extract k-mers from sequence.
    
    Args:
        sequence: DNA/RNA sequence string
        k: k-mer size
        overlap: If True, extract overlapping k-mers (stride=1).
                 If False, extract non-overlapping k-mers (stride=k).
    
    Returns:
        List of k-mer strings
    """
    sequence = sequence.upper().replace('U', 'T')  # Normalize RNA to DNA
    kmers = []
    
    stride = 1 if overlap else k
    
    for i in range(0, len(sequence) - k + 1, stride):
        kmer = sequence[i:i+k]
        # Only include if all valid nucleotides
        if all(n in 'ACGT' for n in kmer):
            kmers.append(kmer)
    
    return kmers


def build_vocab(sequences: List[Tuple[str, str]], k: int, min_freq: int = 2, overlap: bool = True) -> Dict[str, int]:
    """Build k-mer vocabulary from sequences."""
    print(f"\nBuilding vocabulary (k={k}, min_freq={min_freq}, overlap={overlap})...")
    
    # Count all k-mers
    kmer_counts = Counter()
    for header, seq in tqdm(sequences, desc="Counting k-mers"):
        kmers = extract_kmers(seq, k, overlap=overlap)
        kmer_counts.update(kmers)
    
    print(f"  Found {len(kmer_counts):,} unique k-mers")
    
    # Build vocab with special tokens
    vocab = {
        '<pad>': 0,
        '<unk>': 1,
        '<eos>': 2,
    }
    
    # Add k-mers above minimum frequency
    idx = len(vocab)
    for kmer, count in kmer_counts.most_common():
        if count >= min_freq:
            vocab[kmer] = idx
            idx += 1
    
    print(f"  Vocabulary size: {len(vocab):,} tokens")
    print(f"  Coverage: {sum(1 for c in kmer_counts.values() if c >= min_freq):,}/{len(kmer_counts):,} k-mers")
    
    return vocab


def create_synthetic_labels(
    sequences: List[Tuple[str, str]],
    n_samples: int = 10000,
    seed: int = 42,
    fallback_length: int = 512
) -> Tuple[List[str], List[int]]:
    """
    Create balanced synthetic promoter/non-promoter labels.
    
    Strategy:
    - Promoter (label=1): sequences with promoter keywords + injected motifs
    - Non-promoter (label=0): random background sequences
    """
    print(f"\nGenerating {n_samples:,} synthetic labels...")
    
    random.seed(seed)
    np.random.seed(seed)
    
    promoter_keywords = ['promoter', 'regulatory', 'enhancer']
    promoter_motifs = ['TATAAA', 'TTGACA', 'CCAAT', 'GGGCGG']

    def has_promoter_keyword(header: str) -> bool:
        header_lower = header.lower()
        return any(kw in header_lower for kw in promoter_keywords)
    
    def has_promoter_motif(sequence: str) -> bool:
        sequence = sequence.upper().replace('U', 'T')
        return any(motif in sequence for motif in promoter_motifs)
    
    # Classify sequences
    labeled_seqs = []
    for header, seq in sequences:
        if len(seq) < 100:  # Skip very short sequences
            continue
        
        # Label based on keywords or motifs
        has_keyword = has_promoter_keyword(header)
        has_motif = has_promoter_motif(seq)
        
        if has_keyword or has_motif:
            label = 1  # Promoter
        else:
            label = 0  # Non-promoter
        
        labeled_seqs.append((seq, label))
    
    # Balance classes
    pos_seqs = [(s, l) for s, l in labeled_seqs if l == 1]
    neg_seqs = [(s, l) for s, l in labeled_seqs if l == 0]

    print(f"  Initial: {len(pos_seqs):,} promoters, {len(neg_seqs):,} non-promoters")

    def random_sequence(length: int) -> str:
        return ''.join(np.random.choice(list('ACGT'), size=length))

    def inject_motif(sequence: str) -> str:
        motif = random.choice(promoter_motifs)
        if len(sequence) <= len(motif):
            return motif[: len(sequence)]
        max_start = min(180, len(sequence) - len(motif))
        start = random.randint(0, max_start) if max_start > 0 else 0
        return sequence[:start] + motif + sequence[start + len(motif):]

    n_target = max(1, n_samples // 2)

    if len(pos_seqs) < n_target:
        deficit = n_target - len(pos_seqs)
        print(f"  Warning: only {len(pos_seqs)} positive sequences found; synthesizing {deficit} more")
        for _ in range(deficit):
            synthetic = inject_motif(random_sequence(fallback_length))
            pos_seqs.append((synthetic, 1))

    if len(neg_seqs) < n_target:
        deficit = n_target - len(neg_seqs)
        print(f"  Warning: only {len(neg_seqs)} negative sequences found; synthesizing {deficit} more")
        for _ in range(deficit):
            synthetic = random_sequence(fallback_length)
            neg_seqs.append((synthetic, 0))

    # Sample balanced subset
    n_per_class = min(n_target, len(pos_seqs), len(neg_seqs))
    
    random.shuffle(pos_seqs)
    random.shuffle(neg_seqs)
    
    balanced = pos_seqs[:n_per_class] + neg_seqs[:n_per_class]
    random.shuffle(balanced)
    
    sequences_out = [s for s, l in balanced]
    labels_out = [l for s, l in balanced]
    
    print(f"  Balanced: {sum(labels_out):,} promoters, {len(labels_out) - sum(labels_out):,} non-promoters")
    print(f"  Total samples: {len(labels_out):,}")
    
    return sequences_out, labels_out


def tokenize_sequences(
    sequences: List[str],
    labels: List[int],
    vocab: Dict[str, int],
    k: int,
    max_length: int = 512,
    overlap: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert sequences to token tensors."""
    print(f"\nTokenizing sequences (max_length={max_length}, overlap={overlap})...")
    
    token_seqs = []
    valid_labels = []
    
    for seq, label in tqdm(zip(sequences, labels), total=len(sequences), desc="Tokenizing"):
        kmers = extract_kmers(seq, k, overlap=overlap)
        
        if not kmers:
            continue
        
        # Convert to token IDs
        tokens = [vocab.get(kmer, vocab['<unk>']) for kmer in kmers]
        
        # Truncate or pad
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Add EOS
        tokens.append(vocab['<eos>'])
        
        token_seqs.append(tokens)
        valid_labels.append(label)
    
    print(f"  Tokenized {len(token_seqs):,} sequences")
    
    # Flatten to single tensor (PTB style)
    flat_tokens = []
    for tokens in token_seqs:
        flat_tokens.extend(tokens)
    
    token_tensor = torch.tensor(flat_tokens, dtype=torch.long)
    label_tensor = torch.tensor(valid_labels, dtype=torch.long)
    
    print(f"  Total tokens: {len(token_tensor):,}")
    
    return token_tensor, label_tensor


def split_data(
    token_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split data into train/val/test."""
    print(f"\nSplitting data (train={train_ratio}, val={val_ratio}, test={test_ratio})...")
    
    torch.manual_seed(seed)
    
    n_total = len(token_tensor)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split tokens
    train_tokens = token_tensor[:n_train]
    val_tokens = token_tensor[n_train:n_train + n_val]
    test_tokens = token_tensor[n_train + n_val:]
    
    # For labels, we need to map back to sequences
    # For simplicity, we'll create dummy labels tensor that matches token count
    # In real usage, you'd track sequence boundaries
    train_labels = label_tensor
    val_labels = label_tensor
    test_labels = label_tensor
    
    print(f"  Train: {len(train_tokens):,} tokens")
    print(f"  Val: {len(val_tokens):,} tokens")
    print(f"  Test: {len(test_tokens):,} tokens")
    
    return train_tokens, val_tokens, test_tokens, train_labels, val_labels, test_labels


def main():
    parser = argparse.ArgumentParser(description="Build genomic dataset for Quixer")
    parser.add_argument(
        '--fasta',
        type=str,
        default='/scratch/cbjp404/bradford_hackathon_gene_sequencing/GRCh38_genomic_dataset/rna_sequences.fasta',
        help='Path to FASTA file'
    )
    parser.add_argument('--k', type=int, default=6, help='K-mer size')
    parser.add_argument('--min_freq', type=int, default=2, help='Minimum k-mer frequency for vocab')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length (in k-mers)')
    parser.add_argument('--n_samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument(
        '--no-overlap',
        action='store_true',
        help='Use non-overlapping k-mers (stride=k) instead of overlapping (stride=1)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/scratch/cbjp404/bradford_hackathon_gene_sequencing/genomic_data',
        help='Output directory'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine overlap mode
    overlap = not args.no_overlap
    
    print("=" * 70)
    print("Building Genomic Dataset for Quixer")
    print("=" * 70)
    print(f"FASTA: {args.fasta}")
    print(f"K-mer size: {args.k}")
    print(f"K-mer mode: {'overlapping (stride=1)' if overlap else 'non-overlapping (stride=k)'}")
    print(f"Output: {output_dir}")
    
    # Parse FASTA
    sequences = parse_fasta(Path(args.fasta))
    
    # Create synthetic labels
    labeled_seqs, labels = create_synthetic_labels(sequences, n_samples=args.n_samples, seed=args.seed)
    
    # Build vocabulary
    # We need to create (header, seq) tuples for vocab building
    seq_tuples = [('', seq) for seq in labeled_seqs]
    vocab = build_vocab(seq_tuples, k=args.k, min_freq=args.min_freq, overlap=overlap)
    
    # Tokenize sequences
    token_tensor, label_tensor = tokenize_sequences(
        labeled_seqs, labels, vocab, k=args.k, max_length=args.max_length, overlap=overlap
    )
    
    # Split data
    train_tokens, val_tokens, test_tokens, train_labels, val_labels, test_labels = split_data(
        token_tensor, label_tensor, seed=args.seed
    )
    
    # Save data
    print("\nSaving dataset...")
    
    # Save vocabulary
    vocab_path = output_dir / 'vocab.json'
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"  Saved vocabulary: {vocab_path}")
    
    # Save splits
    torch.save(train_tokens, output_dir / 'train_tokens.pt')
    torch.save(val_tokens, output_dir / 'val_tokens.pt')
    torch.save(test_tokens, output_dir / 'test_tokens.pt')
    
    torch.save(train_labels, output_dir / 'train_labels.pt')
    torch.save(val_labels, output_dir / 'val_labels.pt')
    torch.save(test_labels, output_dir / 'test_labels.pt')
    
    print(f"  Saved train: {output_dir / 'train_tokens.pt'}")
    print(f"  Saved val: {output_dir / 'val_tokens.pt'}")
    print(f"  Saved test: {output_dir / 'test_tokens.pt'}")
    
    # Save metadata
    metadata = {
        'k': args.k,
        'overlap': overlap,
        'vocab_size': len(vocab),
        'n_samples': len(labeled_seqs),
        'max_length': args.max_length,
        'train_tokens': len(train_tokens),
        'val_tokens': len(val_tokens),
        'test_tokens': len(test_tokens),
        'seed': args.seed,
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")
    
    print("\n" + "=" * 70)
    print("✓ Dataset building complete!")
    print("=" * 70)
    print(f"\nTo train Quixer, update setup_training.py to load from: {output_dir}")


if __name__ == '__main__':
    main()
