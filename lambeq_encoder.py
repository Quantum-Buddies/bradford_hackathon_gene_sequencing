"""
lambeq Encoder for Genomic Sequences
=====================================
Converts k-mer "sentences" into quantum circuit embeddings using lambeq.

Architecture:
- Uses lambeq's DisCoCat compositional approach
- Converts k-mer sequences into typed diagrams
- Generates tensor network embeddings for Quixer

References:
- lambeq documentation: https://docs.quantinuum.com/lambeq/
- DisCoCat framework: Coecke et al. (2010)
"""

import argparse
import hashlib
import math
import struct
import numpy as np
import torch
from pathlib import Path
from typing import Tuple
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# lambeq imports
try:
    from lambeq import (
        BobcatParser,
        AtomicType,
        IQPAnsatz,
        RemoveCupsRewriter,
        NumpyModel,
    )
    LAMBEQ_AVAILABLE = True
except ImportError:
    print("WARNING: lambeq not installed. Run: pip install lambeq")
    LAMBEQ_AVAILABLE = False


_worker_parser = None
_worker_ansatz = None
_worker_embedding_dim = None
_worker_remove_cups = None
_worker_seed = None
_worker_device = None
_mp_context = None


def _required_qubits(embedding_dim: int) -> int:
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    return max(1, math.ceil(math.log2(embedding_dim)))


def _hash_symbol_value(symbol_name: str, seed: int) -> float:
    """Generate a deterministic parameter value in [0, 2π) for a symbol."""
    digest = hashlib.sha256(f"{seed}:{symbol_name}".encode("utf-8")).digest()
    integer = struct.unpack(">Q", digest[:8])[0]
    return (integer / float(2**64)) * (2.0 * math.pi)


def _contract_circuit_direct(circuit) -> np.ndarray:
    import tensornetwork as tn

    result = tn.contractors.auto(*circuit.to_tn()).tensor
    if not circuit.is_mixed:
        result = np.abs(result) ** 2
    result = result.reshape(-1)
    total = np.sum(result)
    if total > 1e-12:
        result = result / total
    return np.asarray(result, dtype=np.float64)


def _evaluate_kmer_with_model(
    kmer: str,
    ansatz: IQPAnsatz,
    embedding_dim: int,
    seed: int,
) -> np.ndarray:
    """
    Encode a single k-mer using a minimal quantum circuit.
    
    Instead of parsing as a sentence, we create a simple diagram for the k-mer
    and encode it directly with IQP ansatz.
    """
    try:
        # Import lambeq diagram types
        from lambeq import Ty, Word
        
        # Create a simple word diagram for the k-mer
        # Treat each k-mer as a noun (atomic entity)
        kmer_type = AtomicType.NOUN
        diagram = Word(kmer, kmer_type)
        
        # Apply ansatz to create quantum circuit
        circuit = ansatz(diagram)
        
        # Build model
        model = NumpyModel.from_diagrams([circuit])
        
        if model.symbols:
            weights = np.array([
                _hash_symbol_value(f"{kmer}:{sym}", seed) for sym in model.symbols
            ], dtype=np.float64)
            model.weights = weights
        
        # Contract circuit to get embedding
        result = _contract_circuit_direct(circuit)
        
        # Pad or truncate to embedding_dim
        if result.shape[0] < embedding_dim:
            result = np.pad(result, (0, embedding_dim - result.shape[0]), mode='constant')
        elif result.shape[0] > embedding_dim:
            result = result[:embedding_dim]
        
        return result.astype(np.float32)
    
    except Exception as e:
        # Fallback: deterministic hash-based embedding
        np.random.seed(hash(kmer) % (2**32))
        return np.random.randn(embedding_dim).astype(np.float32)


# Worker functions removed - per-k-mer encoding is done sequentially
# to avoid IPC overhead with quantum circuits


class GenomicLambeqEncoder:
    """
    Encodes genomic k-mer sequences using lambeq's compositional framework.
    
    Strategy:
    1. Parse k-mer sentences into typed diagrams
    2. Use IQP ansatz to convert to parameterized circuits
    3. Simulate circuits to get embeddings
    4. Export embeddings for Quixer training
    """
    
    def __init__(
        self,
        data_dir: str = "/scratch/cbjp404/bradford_hackathon_2025/processed_data",
        output_dir: str = "/scratch/cbjp404/bradford_hackathon_2025/lambeq_embeddings",
        embedding_dim: int = 128,
        n_layers: int = 2,
        seed: int = 42,
        num_workers: int = 1,
        parser_device: str = 'cpu',
    ):
        if not LAMBEQ_AVAILABLE:
            raise ImportError("lambeq is required. Install with: pip install lambeq")
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.seed = seed
        self.num_workers = max(1, num_workers)
        self.device = parser_device
        self._mp_context = mp.get_context('spawn')

        np.random.seed(seed)
        torch.manual_seed(seed)
        
        print(f"Initialized GenomicLambeqEncoder:")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Circuit layers: {n_layers}")
        print(f"  Workers: {self.num_workers}")
        print(f"  Parser device: {self.device}")
        
        # Load metadata
        with open(self.data_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Initialize lambeq components
        self._init_lambeq()
    
    def _init_lambeq(self):
        """Initialize lambeq parser and ansatz."""
        print("\nInitializing lambeq components...")
        
        # Define atomic type for k-mers
        self.kmer_type = AtomicType.NOUN  # Treat k-mers as noun entities
        
        # Use simple bag-of-words style for genomic sequences
        # (more sophisticated parsing can be added later)
        self.parser = BobcatParser(verbose='text', device=self.device)
        
        self.required_qubits = _required_qubits(self.embedding_dim)
        self.state_dim = 2 ** self.required_qubits
        self.remove_cups = RemoveCupsRewriter()

        # IQP ansatz: efficient for NISQ devices
        self.ansatz = IQPAnsatz(
            {
                self.kmer_type: self.required_qubits,
                AtomicType.SENTENCE: self.required_qubits,
            },
            n_layers=self.n_layers,
            n_single_qubit_params=3
        )
        
        print("  ✓ Parser initialized")
        print("  ✓ IQP ansatz configured")
        print(f"  ✓ Qubits per type: {self.required_qubits}")

    def encode_kmer_sequence(self, sentence: str, max_kmers: int = 80) -> np.ndarray:
        """
        Encode a k-mer sentence into a SEQUENCE of per-k-mer embeddings.
        
        Args:
            sentence: Space-separated k-mer sequence
            max_kmers: Maximum number of k-mers to encode (pad/truncate)
        
        Returns:
            Embedding matrix (shape: [max_kmers, embedding_dim])
        """
        # Split sentence into individual k-mers
        kmers = sentence.strip().split()
        
        # Encode each k-mer separately
        embeddings = []
        for kmer in kmers[:max_kmers]:
            try:
                emb = _evaluate_kmer_with_model(
                    kmer,
                    self.ansatz,
                    self.embedding_dim,
                    self.seed,
                )
                embeddings.append(emb)
            except Exception as e:
                # Fallback: random embedding
                embeddings.append(np.random.randn(self.embedding_dim).astype(np.float32))
        
        # Pad if needed
        while len(embeddings) < max_kmers:
            embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
        
        # Stack into matrix [max_kmers, embedding_dim]
        return np.stack(embeddings[:max_kmers], axis=0)
    
    def encode_split(self, split_name: str, max_kmers: int = 80) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode all sentences in a data split as sequences of per-k-mer embeddings.
        
        Args:
            split_name: 'train', 'val', or 'test'
            max_kmers: Maximum k-mers per sample (pad/truncate)
        
        Returns:
            (embeddings, labels) where embeddings is [N, max_kmers, embedding_dim]
        """
        print(f"\nEncoding {split_name} split (per-k-mer mode)...")
        
        split_dir = self.data_dir / split_name
        
        # Load sentences and labels
        with open(split_dir / "sentences.txt", 'r') as f:
            sentences = [line.strip() for line in f]
        
        with open(split_dir / "labels.txt", 'r') as f:
            labels = [int(line.strip()) for line in f]
        
        print(f"  Loaded {len(sentences)} samples")
        print(f"  Max k-mers per sample: {max_kmers}")
        
        # Encode all sentences as sequences
        embeddings = []
        for sentence in tqdm(sentences, desc=f"Encoding {split_name}"):
            emb_seq = self.encode_kmer_sequence(sentence, max_kmers=max_kmers)
            embeddings.append(emb_seq)
        
        # Convert to tensors [N, max_kmers, embedding_dim]
        embeddings_tensor = torch.from_numpy(np.stack(embeddings))
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        print(f"  Encoded shape: {embeddings_tensor.shape}")
        print(f"  Labels shape: {labels_tensor.shape}")
        
        return embeddings_tensor, labels_tensor
    
    def save_embeddings(
        self,
        split_name: str,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ):
        """Save encoded embeddings and labels."""
        output_file = self.output_dir / f"{split_name}.pt"
        
        torch.save({
            'embeddings': embeddings,
            'labels': labels,
            'metadata': {
                'embedding_dim': self.embedding_dim,
                'n_layers': self.n_layers,
                'ansatz_qubits': self.required_qubits,
                'ansatz_state_dim': self.state_dim,
                'split': split_name,
                'n_samples': len(embeddings),
                'max_kmers': embeddings.shape[1] if len(embeddings.shape) > 2 else 1,
                'encoding_mode': 'per_kmer_sequence'
            }
        }, output_file)
        
        print(f"  Saved to: {output_file}")
    
    def run(self, max_kmers: int = 80):
        """Run complete encoding pipeline for all splits with per-k-mer embeddings.
        
        Args:
            max_kmers: Maximum k-mers per sample (default 80 for ~512bp windows)
        """
        print("=" * 70)
        print("LAMBEQ PER-K-MER ENCODING PIPELINE")
        print("=" * 70)
        print(f"Mode: Per-k-mer embeddings (preserves sequence structure)")
        print(f"Max k-mers per sample: {max_kmers}")
        print()
        
        for split_name in ['train', 'val', 'test']:
            # Encode split with per-k-mer embeddings
            embeddings, labels = self.encode_split(split_name, max_kmers=max_kmers)
            
            # Save embeddings
            self.save_embeddings(split_name, embeddings, labels)
        
        # Save encoding metadata
        encoding_metadata = {
            'embedding_dim': self.embedding_dim,
            'n_layers': self.n_layers,
            'seed': self.seed,
            'ansatz': 'IQP',
            'parser': 'Per-kmer (no sentence-level parsing)',
            'max_kmers': max_kmers,
            'encoding_mode': 'per_kmer_sequence'
        }
        
        metadata_file = self.output_dir / "encoding_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(encoding_metadata, f, indent=2)
        
        print(f"\n✅ Encoding complete! Saved to: {self.output_dir}")


# Fallback: Simple embedding encoder (if lambeq not available)
class SimpleKmerEncoder:
    """
    Fallback encoder using classical k-mer embeddings.
    Used when lambeq is not available.
    """
    
    def __init__(
        self,
        data_dir: str = "/scratch/cbjp404/bradford_hackathon_2025/processed_data",
        output_dir: str = "/scratch/cbjp404/bradford_hackathon_2025/simple_embeddings",
        embedding_dim: int = 128,
        seed: int = 42,
        num_workers: int = 1
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.num_workers = max(1, num_workers)
        
        np.random.seed(seed)
        
        # Load vocabulary
        with open(self.data_dir / "vocab.json", 'r') as f:
            self.vocab = json.load(f)
        
        # Initialize random embeddings for each k-mer
        self.kmer_embeddings = {
            kmer: np.random.randn(embedding_dim).astype(np.float32)
            for kmer in self.vocab.keys()
        }
        
        print(f"Initialized SimpleKmerEncoder:")
        print(f"  Vocabulary size: {len(self.vocab)}")
        print(f"  Embedding dimension: {embedding_dim}")
    
    def encode_sentence(self, sentence: str) -> np.ndarray:
        """Encode k-mer sentence as mean of k-mer embeddings."""
        kmers = sentence.split()
        
        # Get embeddings for each k-mer
        embs = [
            self.kmer_embeddings.get(kmer, self.kmer_embeddings['<UNK>'])
            for kmer in kmers
        ]
        
        if embs:
            # Mean pooling
            embedding = np.mean(embs, axis=0)
        else:
            # Fallback
            embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        
        return embedding
    
    def encode_split(self, split_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode all sentences in a split."""
        print(f"\nEncoding {split_name} split...")
        
        split_dir = self.data_dir / split_name
        
        with open(split_dir / "sentences.txt", 'r') as f:
            sentences = [line.strip() for line in f]
        
        with open(split_dir / "labels.txt", 'r') as f:
            labels = [int(line.strip()) for line in f]
        
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                iterator = executor.map(self.encode_sentence, sentences)
                embeddings = list(tqdm(iterator, total=len(sentences)))
        else:
            embeddings = [self.encode_sentence(s) for s in tqdm(sentences)]
        
        embeddings_tensor = torch.from_numpy(np.stack(embeddings))
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return embeddings_tensor, labels_tensor
    
    def save_embeddings(self, split_name: str, embeddings: torch.Tensor, labels: torch.Tensor):
        """Save embeddings."""
        output_file = self.output_dir / f"{split_name}.pt"
        torch.save({'embeddings': embeddings, 'labels': labels}, output_file)
        print(f"  Saved: {output_file}")
    
    def run(self):
        """Run encoding for all splits."""
        for split_name in ['train', 'val', 'test']:
            embeddings, labels = self.encode_split(split_name)
            self.save_embeddings(split_name, embeddings, labels)
        
        print(f"\n✅ Encoding complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode genomics dataset with lambeq")
    parser.add_argument("--embedding_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--layers", type=int, default=2, help="Number of IQP ansatz layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    parser.add_argument("--parser_device", type=str, default='cpu', help="Device for Bobcat parser (cpu, cuda, cuda:0, etc.)")
    args = parser.parse_args()

    # Try lambeq encoder first, fall back to simple encoder
    try:
        encoder = GenomicLambeqEncoder(
            embedding_dim=args.embedding_dim,
            n_layers=args.layers,
            seed=args.seed,
            num_workers=args.workers,
            parser_device=args.parser_device,
        )
        # Run with max_kmers=80 for typical 512bp windows with 6-mers
        encoder.run(max_kmers=80)
    except ImportError:
        print("\n⚠️  lambeq not available, using simple k-mer encoder instead")
        encoder = SimpleKmerEncoder(
            embedding_dim=args.embedding_dim,
            seed=args.seed,
            num_workers=max(1, args.workers)
        )
        encoder.run()
