import json
import random
from pathlib import Path
import bisect
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class GenomicPredictionDataset(Dataset):
    """Memory-efficient dataset for genomic next-base prediction.

    This dataset loads a large, pre-processed tensor of concatenated tokenized
    sequences using memory-mapping (mmap), avoiding high RAM usage. It calculates
    valid sliding window indices on initialization, ensuring that windows do not
    cross the boundaries of the original sequences.
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int,
        stride: int = 1,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        """
        Args:
            data_dir (str): Directory containing 'all_tokens.pt' and 'metadata.json'.
            window_size (int): The size of the input sequence window.
            stride (int): The step size to slide the window across tokens.
            split (str): The dataset split to use: 'train', 'val', or 'test'.
            train_ratio (float): Fraction of the total data to use for training.
            val_ratio (float): Fraction of the total data to use for validation.
            seed (int): Random seed for shuffling and splitting the data.
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride

        # Load metadata
        print(f"Loading metadata for '{split}' split...")
        with open(self.data_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        self.seq_lengths = metadata['seq_lengths']
        self.total_tokens = metadata['total_tokens']
        self.n_sequences = len(self.seq_lengths)

        # Load the entire token dataset using memory-mapping
        # This avoids loading the whole file into RAM.
        print("Memory-mapping token data...")
        self.tokens = torch.load(self.data_dir / 'all_tokens.pt', mmap=True)

        # Precompute start positions of each sequence in the concatenated tensor
        self.seq_start_positions = []
        current_pos = 0
        for length in self.seq_lengths:
            self.seq_start_positions.append(current_pos)
            current_pos += length

        # Determine sequence-level splits (same shuffle for all splits via seed)
        print("Preparing sequence-level splits...")
        all_seq_indices = list(range(self.n_sequences))
        rng = random.Random(seed)
        rng.shuffle(all_seq_indices)

        train_size = int(self.n_sequences * train_ratio)
        val_size = int(self.n_sequences * val_ratio)

        train_seq_indices = all_seq_indices[:train_size]
        val_seq_indices = all_seq_indices[train_size : train_size + val_size]
        test_seq_indices = all_seq_indices[train_size + val_size :]

        if split == 'train':
            self.seq_indices = train_seq_indices
        elif split == 'val':
            self.seq_indices = val_seq_indices
        elif split == 'test':
            self.seq_indices = test_seq_indices
        else:
            raise ValueError("split must be one of 'train', 'val', or 'test'")

        # For sequences in this split, compute how many windows each contributes
        print("Calculating windows per sequence...")
        self.windows_per_seq = []
        for seq_idx in self.seq_indices:
            length = self.seq_lengths[seq_idx]
            max_start = length - self.window_size - 1
            if max_start >= 0:
                n_windows = max_start // self.stride + 1
            else:
                n_windows = 0
            self.windows_per_seq.append(n_windows)

        # Prefix sums over windows to map a flat index -> (sequence, offset)
        self.cum_windows = []
        total_windows = 0
        for n in self.windows_per_seq:
            self.cum_windows.append(total_windows)
            total_windows += n

        self.total_windows = total_windows

        print(f"  - Found {self.total_windows:,} samples for '{split}' split.")

    def __len__(self) -> int:
        """Return the total number of samples in this split."""
        return self.total_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one sample: (input_window, next_base)."""
        if idx < 0 or idx >= self.total_windows:
            raise IndexError("Index out of range")

        # Find which sequence this sample belongs to via prefix sums
        seq_pos = bisect.bisect_right(self.cum_windows, idx) - 1
        if seq_pos < 0:
            seq_pos = 0

        # Global sequence index
        seq_idx = self.seq_indices[seq_pos]

        # Offset of this window within the chosen sequence
        window_offset = idx - self.cum_windows[seq_pos]

        # Compute the starting token index in the concatenated tensor
        seq_start_token = self.seq_start_positions[seq_idx]
        start_pos = seq_start_token + window_offset * self.stride

        input_window = self.tokens[start_pos : start_pos + self.window_size].long()
        target_base = self.tokens[start_pos + self.window_size].long()

        return input_window, target_base
