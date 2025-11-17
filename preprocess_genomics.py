"""
Genomic Data Preprocessing for Quixer + lambeq
===============================================
Prepares GRCh38 genomic sequences for quantum NLP classification.

Based on research insights from:
- QNLP applications to bioinformatics (Frontiers, 2025)
- k-mer based genomics ML (iMOKA, Genome Biology 2020)
- DNABERT-2 tokenization strategies (arXiv 2023)

Task: Binary classification of transcript type (coding mRNA vs. non-coding RNA)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import json
from collections import Counter
from sklearn.model_selection import train_test_split
from itertools import product

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
        k: int = 4,
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
        self._kmer_classes: List[str] = []
        self.label_lookup: Dict[str, int] = {}
        self.chunked_windows: List[List[str]] = []
        
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

        if not rna_file.exists():
            print("  WARNING: RNA summary file not found. Falling back to synthetic descriptions.")
            synthetic_desc = (
                ["synthetic coding mRNA transcript"] * 50
                + ["synthetic noncoding lncRNA transcript"] * 50
            )
            return pd.DataFrame({'Description': synthetic_desc})
        
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
    
    def chunk_sequence(self, sequence: str, chunk_size: int = None) -> List[str]:
        """
        Break a sequence into non-overlapping chunks of length `chunk_size`.
        """
        if chunk_size is None:
            chunk_size = self.k
        if chunk_size <= 0:
            return []
        
        sequence = sequence.upper().replace('U', 'T')
        chunks = []
        for i in range(0, len(sequence) - chunk_size + 1, chunk_size):
            chunk = sequence[i:i+chunk_size]
            if len(chunk) == chunk_size and all(n in 'ACGT' for n in chunk):
                chunks.append(chunk)
        return chunks
    
    def _get_all_kmer_classes(self) -> List[str]:
        """Return cached list of all possible k-mer combinations."""
        if not self._kmer_classes:
            self._kmer_classes = [''.join(p) for p in product('ACGT', repeat=self.k)]
        return self._kmer_classes
    
    def create_synthetic_labels(
        self,
        df: pd.DataFrame,
        n_samples: int = 10000
    ) -> Tuple[List[str], List[int]]:
        """
        Create labeled non-overlapping k-mer sequences that cover every
        possible nucleotide combination of length `self.k`.

        Strategy:
        - infer coding vs non-coding descriptions to seed realistic motifs
        - synthesize random windows with light motif injection
        - break each window into non-overlapping k-mer "chains"
        - label every k-mer using the complete combination list (ACGT)^k
        - inject any missing k-mers so every label is represented at least once
        """
        print(f"\nGenerating {n_samples} labeled windows (coding vs non-coding, balanced)...")

        # Normalize description column
        desc = df.get('Description') if 'Description' in df.columns else None
        if desc is None:
            desc = pd.Series([''] * len(df))
        desc = desc.fillna('').astype(str)

        def classify(s: str) -> int:
            s_low = s.lower()
            nc_terms = [
                'lncrna', 'ncrna', 'mirna', 'snrna', 'snorna', 'trna', 'rrna',
                'pirna', 'scarna', 'y rna', 'y_rna', 'antisense', 'pseudogene',
                'ribozyme', 'vault', 'misc rna', 'misc_rna'
            ]
            if any(t in s_low for t in nc_terms):
                return 0
            if 'mrna' in s_low:
                return 1
            return 0

        labels_by_row = desc.apply(classify).values
        df_pos = df[labels_by_row == 1]   # coding
        df_neg = df[labels_by_row == 0]   # non-coding

        n_pos = n_samples // 2
        n_neg = n_samples - n_pos

        if len(df_pos) == 0 or len(df_neg) == 0:
            raise RuntimeError("Insufficient rows to build both classes from RNA summary.")

        # Motif-based synthetic sequences to create learnable but non-trivial signal
        rng = np.random.default_rng(self.seed)
        coding_codons = ["ATG", "GCT", "GCC", "GCA", "GCG", "GAA", "GAG", "GGT", "GGC", "GGA", "GGG"]
        stop_codons = ["TAA", "TAG", "TGA"]
        noncoding_motifs = ["AAAAAA", "TTTTTT", "CCCCCC", "GGGGGG"]

        def insert_motif(seq: str, motif: str) -> str:
            i = int(rng.integers(0, max(1, len(seq) - len(motif) + 1)))
            return seq[:i] + motif + seq[i+len(motif):]

        def gen_seq_with_label(label: int) -> str:
            seq = ''.join(rng.choice(list('ACGT'), size=self.window_size))
            if label == 1:
                seq = insert_motif(seq, 'ATG')
                seq = insert_motif(seq, rng.choice(stop_codons))
                for _ in range(3):
                    seq = insert_motif(seq, rng.choice(coding_codons))
            else:
                for _ in range(2):
                    seq = insert_motif(seq, rng.choice(noncoding_motifs))
                seq = seq.replace('ATG', 'ATA')
            return seq

        pos_rows = df_pos.sample(n=n_pos, replace=True, random_state=self.seed)
        neg_rows = df_neg.sample(n=n_neg, replace=True, random_state=self.seed)

        raw_sequences: List[str] = []
        for _ in range(len(pos_rows)):
            raw_sequences.append(gen_seq_with_label(1))
        for _ in range(len(neg_rows)):
            raw_sequences.append(gen_seq_with_label(0))

        indices = np.random.permutation(len(raw_sequences))
        raw_sequences = [raw_sequences[i] for i in indices]

        print(
            f"Generated {len(raw_sequences)} sequences "
            f"(coding={len(pos_rows)}, non-coding={len(neg_rows)})"
        )

        print(f"Breaking down sequences into non-overlapping {self.k}-mers and assigning labels...")
        kmer_classes = self._get_all_kmer_classes()
        self.label_lookup = {kmer: idx for idx, kmer in enumerate(kmer_classes)}
        chunk_sequences: List[str] = []
        chunk_labels: List[int] = []
        seen_counts = Counter()
        chunked_windows: List[List[str]] = []

        for seq in raw_sequences:
            chunks = self.chunk_sequence(seq, chunk_size=self.k)
            if not chunks:
                continue
            chunked_windows.append(chunks)
            for chunk in chunks:
                chunk_sequences.append(chunk)
                label = self.label_lookup[chunk]
                chunk_labels.append(label)
                seen_counts[chunk] += 1

        missing = [kmer for kmer in self.label_lookup if kmer not in seen_counts]
        if missing:
            print(f"  Injecting {len(missing)} missing k-mers to cover all label combinations.")
            for kmer in missing:
                chunk_sequences.append(kmer)
                chunk_labels.append(self.label_lookup[kmer])
                seen_counts[kmer] += 1

        print(f"Total chunked sequences: {len(chunk_sequences)}")
        print(f"Unique {self.k}-mer classes observed: {len(seen_counts)} / {len(self.label_lookup)}")

        self.chunked_windows = chunked_windows
        return chunk_sequences, [int(lbl) for lbl in chunk_labels]
    
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
        """Stratified split into train/val/test sets."""
        sequences = np.array(sequences)
        labels = np.array(labels)

        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio <= 0:
            raise ValueError("Invalid split ratios; ensure train_ratio + val_ratio < 1.0")

        X_train, X_temp, y_train, y_temp = train_test_split(
            sequences, labels, test_size=test_ratio + val_ratio, stratify=labels, random_state=self.seed
        )
        # Split temp into val and test
        rel_test = test_ratio / (test_ratio + val_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=self.seed
        )

        splits = {
            'train': (X_train.tolist(), y_train.tolist()),
            'val': (X_val.tolist(), y_val.tolist()),
            'test': (X_test.tolist(), y_test.tolist()),
        }

        print(f"\nData splits (stratified):")
        for name, (seqs, lbls) in splits.items():
            unique_labels = len(set(lbls))
            print(f"  {name}: {len(seqs)} samples across {unique_labels} classes")
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

        label_map_file = None
        if self.label_lookup:
            label_map_file = self.output_dir / "label_mapping.json"
            with open(label_map_file, 'w') as f:
                json.dump(self.label_lookup, f, indent=2)
            print(f"  Saved label mapping: {label_map_file}")
        
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

        chains_file = None
        if self.chunked_windows:
            chains_file = self.output_dir / "chain_sequences.txt"
            with open(chains_file, 'w') as f:
                for window in self.chunked_windows:
                    if not window:
                        continue
                    f.write(' '.join(window) + '\n')
            print(f"  Saved chunk chain corpus: {chains_file}")
        
        # Save metadata
        if self.label_lookup:
            class_names = list(self.label_lookup.keys())
            n_classes = len(class_names)
        else:
            aggregated_labels = set()
            for _, (_, lbls) in splits.items():
                aggregated_labels.update(lbls)
            n_classes = len(aggregated_labels)
            class_names = [str(lbl) for lbl in sorted(aggregated_labels)]

        metadata = {
            'k': self.k,
            'window_size': self.window_size,
            'stride': self.stride,
            'seed': self.seed,
            'vocab_size': len(vocab),
            'n_classes': n_classes,
            'class_names': class_names,
            'label_map_file': str(label_map_file) if label_map_file else None
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata: {metadata_file}")
        
        print("\n[OK] Preprocessing complete!")

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
        k=4,  # 4-mer tokenization (enumerates every nucleotide combination)
        output_dir=str(Path(__file__).parent / "processed_data"),
        window_size=512,  # 512 bp windows (manageable for quantum circuits)
        stride=256,  # 50% overlap
        seed=42
    )
    
    # Generate 10k samples (adjust as needed)
    preprocessor.run(n_samples=10000)
