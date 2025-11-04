"""
Genomic Data Preprocessing for Quixer + lambeq
===============================================
Prepares GRCh38 genomic sequences for quantum NLP classification.

Based on research insights from:
- QNLP applications to bioinformatics (Frontiers, 2025)
- k-mer based genomics ML (iMOKA, Genome Biology 2020)
- DNABERT-2 tokenization strategies (arXiv 2023)

Task: Binary classification of genomic regions (promoter vs. non-promoter)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import json
from collections import Counter
import re

class GenomicDataPreprocessor:
    """
    Preprocesses GRCh38 genomic data for quantum NLP pipelines.
    
    Strategy:
    1. Extract sequence windows from RNA/genomic summaries
    2. Label promoter regions using annotations
    3. Tokenize using overlapping k-mers
    4. Generate train/val/test splits
    5. Export for lambeq encoding
    """
    
    def __init__(
        self,
        data_dir: str = "/scratch/cbjp404/bradford_hackathon_2025/GRCh38_genomic_dataset",
        output_dir: str = "/scratch/cbjp404/bradford_hackathon_2025/processed_data",
        k: int = 6,
        window_size: int = 512,
        stride: int = 256,
        seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.k = k  # k-mer size
        self.window_size = window_size  # sequence window length
        self.stride = stride  # sliding window stride
        self.seed = seed
        
        np.random.seed(seed)
        
        print(f"Initialized GenomicDataPreprocessor:")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  k-mer size: {k}")
        print(f"  Window size: {window_size} bp")
        print(f"  Stride: {stride} bp")
    
    def load_rna_data(self) -> pd.DataFrame:
        """Load RNA summary data (contains transcript information)."""
        rna_file = self.data_dir / "GRCh38_latest_rna_summary.csv"
        print(f"\nLoading RNA data from {rna_file}...")
        
        # Read with low_memory=False to handle mixed types
        df = pd.read_csv(rna_file, low_memory=False)
        print(f"Loaded {len(df)} RNA records")
        print(f"Columns: {list(df.columns)}")
        return df
    
    def extract_kmers(self, sequence: str, k: int = None) -> List[str]:
        """
        Extract overlapping k-mers from a sequence.
        
        Args:
            sequence: DNA/RNA sequence string
            k: k-mer size (defaults to self.k)
        
        Returns:
            List of k-mer strings
        """
        if k is None:
            k = self.k
        
        sequence = sequence.upper().replace('U', 'T')  # Normalize RNA to DNA
        kmers = []
        
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            # Only include if all valid nucleotides
            if all(n in 'ACGT' for n in kmer):
                kmers.append(kmer)
        
        return kmers
    
    def sequence_to_sentence(self, sequence: str) -> str:
        """
        Convert genomic sequence to space-separated k-mer 'sentence'.
        
        This format enables lambeq to parse it as compositional structure.
        """
        kmers = self.extract_kmers(sequence)
        return ' '.join(kmers)
    
    def create_synthetic_labels(
        self,
        df: pd.DataFrame,
        n_samples: int = 10000
    ) -> Tuple[List[str], List[int]]:
        """
        Create synthetic promoter/non-promoter labels.
        
        Strategy:
        - Promoter (label=1): regions with high gene density indicators
        - Non-promoter (label=0): intergenic or low-annotation regions
        
        Args:
            df: RNA dataframe
            n_samples: number of windows to generate
        
        Returns:
            (sequences, labels) where labels are 0/1
        """
        print(f"\nGenerating {n_samples} synthetic labeled windows...")
        
        sequences = []
        labels = []
        
        # Simple heuristic: look for annotation keywords
        promoter_keywords = ['promoter', 'transcript', 'mrna', 'gene']
        
        for idx, row in df.head(min(len(df), n_samples * 10)).iterrows():
            # Extract description or sequence-related info
            description = str(row.get('Description', '')).lower()
            
            # Generate synthetic sequence (placeholder - in real use, fetch from FASTA)
            # For demonstration, create random sequences
            seq_len = self.window_size
            seq = ''.join(np.random.choice(['A', 'C', 'G', 'T'], size=seq_len))
            
            # Label based on description
            is_promoter = any(kw in description for kw in promoter_keywords)
            label = 1 if is_promoter else 0
            
            sequences.append(seq)
            labels.append(label)
            
            if len(sequences) >= n_samples:
                break
        
        print(f"Generated {len(sequences)} sequences")
        print(f"  Promoters (label=1): {sum(labels)}")
        print(f"  Non-promoters (label=0): {len(labels) - sum(labels)}")
        
        return sequences, labels
    
    def build_vocab(self, sequences: List[str]) -> Dict[str, int]:
        """Build k-mer vocabulary from sequences."""
        print("\nBuilding k-mer vocabulary...")
        
        all_kmers = []
        for seq in sequences:
            all_kmers.extend(self.extract_kmers(seq))
        
        kmer_counts = Counter(all_kmers)
        vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        
        for kmer, count in kmer_counts.most_common():
            if kmer not in vocab:
                vocab[kmer] = len(vocab)
        
        print(f"Vocabulary size: {len(vocab)} k-mers")
        return vocab
    
    def split_data(
        self,
        sequences: List[str],
        labels: List[int],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Dict[str, Tuple[List[str], List[int]]]:
        """Split data into train/val/test sets."""
        n = len(sequences)
        indices = np.random.permutation(n)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        splits = {
            'train': ([sequences[i] for i in train_idx], [labels[i] for i in train_idx]),
            'val': ([sequences[i] for i in val_idx], [labels[i] for i in val_idx]),
            'test': ([sequences[i] for i in test_idx], [labels[i] for i in test_idx])
        }
        
        print(f"\nData splits:")
        for split_name, (seqs, lbls) in splits.items():
            print(f"  {split_name}: {len(seqs)} samples (pos={sum(lbls)}, neg={len(lbls)-sum(lbls)})")
        
        return splits
    
    def save_processed_data(
        self,
        splits: Dict[str, Tuple[List[str], List[int]]],
        vocab: Dict[str, int]
    ):
        """Save processed data and metadata."""
        print(f"\nSaving processed data to {self.output_dir}...")
        
        # Save vocabulary
        vocab_file = self.output_dir / "vocab.json"
        with open(vocab_file, 'w') as f:
            json.dump(vocab, f, indent=2)
        print(f"  Saved vocabulary: {vocab_file}")
        
        # Save each split
        for split_name, (sequences, labels) in splits.items():
            split_dir = self.output_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Save sequences as k-mer sentences (for lambeq)
            sentences_file = split_dir / "sentences.txt"
            with open(sentences_file, 'w') as f:
                for seq in sequences:
                    sentence = self.sequence_to_sentence(seq)
                    f.write(sentence + '\n')
            
            # Save labels
            labels_file = split_dir / "labels.txt"
            with open(labels_file, 'w') as f:
                for label in labels:
                    f.write(str(label) + '\n')
            
            print(f"  Saved {split_name}: {len(sequences)} samples")
        
        # Save metadata
        metadata = {
            'k': self.k,
            'window_size': self.window_size,
            'stride': self.stride,
            'seed': self.seed,
            'vocab_size': len(vocab),
            'n_classes': 2,
            'class_names': ['non-promoter', 'promoter']
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata: {metadata_file}")
        
        print("\n✅ Preprocessing complete!")
    
    def run(self, n_samples: int = 10000):
        """Run complete preprocessing pipeline."""
        print("=" * 70)
        print("GENOMIC DATA PREPROCESSING PIPELINE")
        print("=" * 70)
        
        # Load RNA data
        df = self.load_rna_data()
        
        # Create labeled dataset
        sequences, labels = self.create_synthetic_labels(df, n_samples=n_samples)
        
        # Build vocabulary
        vocab = self.build_vocab(sequences)
        
        # Split data
        splits = self.split_data(sequences, labels)
        
        # Save everything
        self.save_processed_data(splits, vocab)


if __name__ == "__main__":
    # Initialize and run preprocessor
    preprocessor = GenomicDataPreprocessor(
        k=6,  # 6-mer tokenization (balances vocabulary size vs. context)
        window_size=512,  # 512 bp windows (manageable for quantum circuits)
        stride=256,  # 50% overlap
        seed=42
    )
    
    # Generate 10k samples (adjust as needed)
    preprocessor.run(n_samples=10000)
