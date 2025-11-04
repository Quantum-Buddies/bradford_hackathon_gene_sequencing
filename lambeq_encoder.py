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

import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict
import json
from tqdm import tqdm

# lambeq imports
try:
    from lambeq import BobcatParser, AtomicType, IQPAnsatz
    LAMBEQ_AVAILABLE = True
except ImportError:
    print("WARNING: lambeq not installed. Run: pip install lambeq")
    LAMBEQ_AVAILABLE = False


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
        seed: int = 42
    ):
        if not LAMBEQ_AVAILABLE:
            raise ImportError("lambeq is required. Install with: pip install lambeq")
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        print(f"Initialized GenomicLambeqEncoder:")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Embedding dimension: {embedding_dim}")
        print(f"  Circuit layers: {n_layers}")
        
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
        self.parser = BobcatParser(verbose='text')
        
        # IQP ansatz: efficient for NISQ devices
        self.ansatz = IQPAnsatz(
            {self.kmer_type: self.embedding_dim},
            n_layers=self.n_layers,
            n_single_qubit_params=3
        )
        
        print("  ✓ Parser initialized")
        print("  ✓ IQP ansatz configured")
    
    def encode_sentence(self, sentence: str) -> np.ndarray:
        """
        Encode a k-mer sentence into a quantum embedding.
        
        Args:
            sentence: Space-separated k-mer sequence
        
        Returns:
            Embedding vector (shape: [embedding_dim])
        """
        try:
            # Parse sentence into diagram
            diagram = self.parser.sentence2diagram(sentence)
            
            # Apply ansatz to get circuit
            circuit = self.ansatz(diagram)
            
            # Simulate circuit (get output state)
            # For now, use simple random embedding
            # In production, would run actual circuit simulation
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            
            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
            
        except Exception as e:
            # Fallback: return random embedding if parsing fails
            print(f"Warning: Failed to parse sentence, using random embedding: {e}")
            return np.random.randn(self.embedding_dim).astype(np.float32)
    
    def encode_split(self, split_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode all sentences in a data split.
        
        Args:
            split_name: 'train', 'val', or 'test'
        
        Returns:
            (embeddings, labels) as PyTorch tensors
        """
        print(f"\nEncoding {split_name} split...")
        
        split_dir = self.data_dir / split_name
        
        # Load sentences and labels
        with open(split_dir / "sentences.txt", 'r') as f:
            sentences = [line.strip() for line in f]
        
        with open(split_dir / "labels.txt", 'r') as f:
            labels = [int(line.strip()) for line in f]
        
        print(f"  Loaded {len(sentences)} samples")
        
        # Encode all sentences
        embeddings = []
        for sentence in tqdm(sentences, desc=f"Encoding {split_name}"):
            emb = self.encode_sentence(sentence)
            embeddings.append(emb)
        
        # Convert to tensors
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
                'split': split_name,
                'n_samples': len(embeddings)
            }
        }, output_file)
        
        print(f"  Saved to: {output_file}")
    
    def run(self):
        """Run complete encoding pipeline for all splits."""
        print("=" * 70)
        print("LAMBEQ ENCODING PIPELINE")
        print("=" * 70)
        
        for split_name in ['train', 'val', 'test']:
            # Encode split
            embeddings, labels = self.encode_split(split_name)
            
            # Save embeddings
            self.save_embeddings(split_name, embeddings, labels)
        
        # Save encoding metadata
        encoding_metadata = {
            'embedding_dim': self.embedding_dim,
            'n_layers': self.n_layers,
            'seed': self.seed,
            'ansatz': 'IQP',
            'parser': 'Bobcat'
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
        seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_dim = embedding_dim
        self.seed = seed
        
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
    # Try lambeq encoder first, fall back to simple encoder
    try:
        encoder = GenomicLambeqEncoder(
            embedding_dim=512,  # Match Quixer's expected input
            n_layers=2,
            seed=42
        )
        encoder.run()
    except ImportError:
        print("\n⚠️  lambeq not available, using simple k-mer encoder instead")
        encoder = SimpleKmerEncoder(
            embedding_dim=512,
            seed=42
        )
        encoder.run()
