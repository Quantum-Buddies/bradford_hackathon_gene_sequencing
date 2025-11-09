#!/usr/bin/env python3
"""Prepare Markov-chain transition tables for classical baselines.

This script reads k-mer sentences (produced by ``prepare_autoregressive_data.py``)
 and builds first-order Markov transition tables over the provided vocabulary.

Outputs (per split):
    - ``{split}_order1.npz`` containing ``transition_counts`` (uint32) with
      shape ``[vocab_size, vocab_size]`` and ``start_counts`` (uint32) with
      shape ``[vocab_size]``.
    - ``markov_metadata.json`` summarising statistics across splits.

These artefacts can be consumed by simple Markov predictors (e.g. maximum-
likelihood next-token prediction).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm


def _load_vocabulary(vocabulary_path: Path) -> Dict[str, int]:
    """Load k-mer vocabulary mapping."""
    with vocabulary_path.open("r") as f:
        payload = json.load(f)

    if "kmer_to_id" in payload:
        return payload["kmer_to_id"]

    # Fallback for legacy vocabularies that directly map token -> id
    return payload


def _infer_vocab_size(kmer_to_id: Dict[str, int]) -> int:
    """Infer vocabulary size from mapping (handles sparse IDs)."""
    if not kmer_to_id:
        raise ValueError("Vocabulary mapping is empty")

    # IDs may not be contiguous; derive size from unique values
    return max(kmer_to_id.values()) + 1


def build_markov_tables(
    data_dir: Path,
    vocabulary_path: Path,
    output_dir: Path,
    order: int = 1,
    smoothing: float = 0.0,
    splits: List[str] | None = None,
) -> None:
    """Build first-order Markov transition tables for each split."""
    if order != 1:
        raise NotImplementedError("Only first-order (order=1) Markov chains are supported.")

    splits = splits or ["train", "val", "test"]
    kmer_to_id = _load_vocabulary(vocabulary_path)
    unk_id = kmer_to_id.get("<UNK>", 0)
    vocab_size = _infer_vocab_size(kmer_to_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: List[dict] = []

    for split in splits:
        sentences_file = data_dir / split / "sentences.txt"
        if not sentences_file.exists():
            print(f"⚠️  Skipping split '{split}' (missing {sentences_file}).")
            continue

        print("=" * 70)
        print(f"BUILDING MARKOV TABLE (split='{split}', order={order})")
        print("=" * 70)
        print(f"Sentences: {sentences_file}")
        print(f"Vocabulary size: {vocab_size}")

        transition_counts = np.zeros((vocab_size, vocab_size), dtype=np.uint32)
        start_counts = np.zeros(vocab_size, dtype=np.uint32)
        total_transitions = 0

        with sentences_file.open("r") as handle:
            for line in tqdm(handle, desc=f"Processing {split}"):
                tokens = line.strip().split()
                if len(tokens) < 2:
                    continue

                token_ids = [kmer_to_id.get(tok, unk_id) for tok in tokens]
                start_counts[token_ids[0]] += 1

                for curr_id, next_id in zip(token_ids[:-1], token_ids[1:]):
                    transition_counts[curr_id, next_id] += 1
                    total_transitions += 1

        row_sums = transition_counts.sum(axis=1, dtype=np.uint64)
        nonzero_transitions = int(np.count_nonzero(transition_counts))

        counts_path = output_dir / f"{split}_order{order}.npz"
        np.savez_compressed(
            counts_path,
            transition_counts=transition_counts,
            start_counts=start_counts,
            row_sums=row_sums,
        )

        entry = {
            "split": split,
            "order": order,
            "counts_file": str(counts_path.name),
            "vocab_size": int(vocab_size),
            "total_transitions": int(total_transitions),
            "nonzero_transitions": nonzero_transitions,
            "smoothing": float(smoothing),
            "smoothing_applied": False,
        }

        if smoothing > 0:
            # Persist smoothed probabilities alongside counts
            probs = transition_counts.astype(np.float32) + smoothing
            denom = probs.sum(axis=1, keepdims=True)
            valid_rows = denom > 0
            probs[valid_rows] /= denom[valid_rows]
            probs[~valid_rows] = 0.0

            probs_path = output_dir / f"{split}_order{order}_probs.npy"
            np.save(probs_path, probs)
            entry.update({
                "probabilities_file": str(probs_path.name),
                "smoothing_applied": True,
            })

        metadata.append(entry)

        print(f"  ✓ Transition counts saved: {counts_path}")
        if entry["smoothing_applied"]:
            print(f"  ✓ Smoothed probabilities saved: {probs_path}")
        print(f"  Total transitions: {total_transitions:,}")
        print(f"  Non-zero entries: {nonzero_transitions:,}")

    if metadata:
        metadata_payload = {
            "data_dir": str(data_dir),
            "vocabulary": str(vocabulary_path),
            "order": order,
            "splits": [m["split"] for m in metadata],
            "entries": metadata,
        }
        metadata_path = output_dir / "markov_metadata.json"
        with metadata_path.open("w") as f:
            json.dump(metadata_payload, f, indent=2)
        print(f"\nMetadata written to {metadata_path}")
    else:
        print("No splits processed; metadata not created.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Markov-chain transition tables from k-mer sentences."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="autoregressive_data",
        help="Directory containing split subfolders with sentences.txt",
    )
    parser.add_argument(
        "--vocabulary",
        type=str,
        default="kmer_vocabulary.json",
        help="Vocabulary JSON mapping k-mers to IDs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="markov_chain_data",
        help="Directory to store Markov transition artefacts",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=1,
        help="Markov chain order (currently only order=1 supported)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.0,
        help="Additive smoothing applied before normalising probabilities",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated list of splits to process",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    vocabulary_path = Path(args.vocabulary)
    output_dir = Path(args.output_dir)
    requested_splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    build_markov_tables(
        data_dir=data_dir,
        vocabulary_path=vocabulary_path,
        output_dir=output_dir,
        order=args.order,
        smoothing=args.smoothing,
        splits=requested_splits,
    )


if __name__ == "__main__":
    main()
