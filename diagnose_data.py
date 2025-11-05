#!/usr/bin/env python3
"""
Diagnostic script to check if the genomics classification task is learnable.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


def load_data(split='train', data_dir='/scratch/cbjp404/bradford_hackathon_2025/processed_data'):
    """Load k-mer sequences and labels."""
    data_dir = Path(data_dir)
    split_dir = data_dir / split
    
    # Load vocabulary
    with open(data_dir / 'vocab.json', 'r') as f:
        vocab = json.load(f)
    
    # Load sentences
    with open(split_dir / 'sentences.txt', 'r') as f:
        sentences = [line.strip() for line in f]
    
    # Load labels
    with open(split_dir / 'labels.txt', 'r') as f:
        labels = [int(line.strip()) for line in f]
    
    return sentences, labels, vocab


def sentences_to_kmer_counts(sentences, vocab):
    """Convert sentences to k-mer count vectors."""
    vocab_size = len(vocab)
    X = np.zeros((len(sentences), vocab_size))
    
    for i, sentence in enumerate(sentences):
        kmers = sentence.split()
        for kmer in kmers:
            if kmer in vocab:
                X[i, vocab[kmer]] += 1
    
    return X


def check_motif_presence(sentences, motifs=['TATAAA', 'TTGACA', 'CCAAT', 'GGGCGG']):
    """Check if promoter motifs are present in first N k-mers."""
    results = {32: [], 64: [], 128: [], 'all': []}
    
    for sentence in sentences:
        kmers = sentence.split()
        
        # Check in first 32 k-mers
        kmers_32 = ' '.join(kmers[:32])
        results[32].append(any(motif in kmers_32 for motif in motifs))
        
        # Check in first 64 k-mers
        kmers_64 = ' '.join(kmers[:64])
        results[64].append(any(motif in kmers_64 for motif in motifs))
        
        # Check in first 128 k-mers
        kmers_128 = ' '.join(kmers[:128])
        results[128].append(any(motif in kmers_128 for motif in motifs))
        
        # Check in all k-mers
        full_sentence = ' '.join(kmers)
        results['all'].append(any(motif in full_sentence for motif in motifs))
    
    return results


def main():
    print("="*70)
    print("GENOMICS DATA DIAGNOSTICS")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    train_sentences, train_labels, vocab = load_data('train')
    val_sentences, val_labels, _ = load_data('val')
    test_sentences, test_labels, _ = load_data('test')
    
    print(f"Train: {len(train_sentences)} samples")
    print(f"Val: {len(val_sentences)} samples")
    print(f"Test: {len(test_sentences)} samples")
    print(f"Vocab size: {len(vocab)}")
    
    # Check label distribution
    print(f"\nLabel distribution:")
    print(f"Train - Class 0: {train_labels.count(0)}, Class 1: {train_labels.count(1)}")
    print(f"Val - Class 0: {val_labels.count(0)}, Class 1: {val_labels.count(1)}")
    print(f"Test - Class 0: {test_labels.count(0)}, Class 1: {test_labels.count(1)}")
    
    # Check sequence lengths
    print(f"\nSequence lengths (number of k-mers):")
    train_lengths = [len(s.split()) for s in train_sentences]
    print(f"Min: {min(train_lengths)}, Max: {max(train_lengths)}, Mean: {np.mean(train_lengths):.1f}")
    
    # Check motif presence
    print(f"\n" + "="*70)
    print("MOTIF PRESENCE ANALYSIS")
    print("="*70)
    print("\nChecking if promoter motifs are present in different sequence windows...")
    
    # Separate positive and negative samples
    pos_sentences = [s for s, l in zip(train_sentences, train_labels) if l == 1]
    neg_sentences = [s for s, l in zip(train_sentences, train_labels) if l == 0]
    
    print(f"\nPositive samples (should contain motifs): {len(pos_sentences)}")
    pos_motifs = check_motif_presence(pos_sentences)
    print(f"  First 32 k-mers: {sum(pos_motifs[32])}/{len(pos_sentences)} ({100*sum(pos_motifs[32])/len(pos_sentences):.1f}%)")
    print(f"  First 64 k-mers: {sum(pos_motifs[64])}/{len(pos_sentences)} ({100*sum(pos_motifs[64])/len(pos_sentences):.1f}%)")
    print(f"  First 128 k-mers: {sum(pos_motifs[128])}/{len(pos_sentences)} ({100*sum(pos_motifs[128])/len(pos_sentences):.1f}%)")
    print(f"  All k-mers: {sum(pos_motifs['all'])}/{len(pos_sentences)} ({100*sum(pos_motifs['all'])/len(pos_sentences):.1f}%)")
    
    print(f"\nNegative samples (should NOT contain motifs): {len(neg_sentences)}")
    neg_motifs = check_motif_presence(neg_sentences)
    print(f"  First 32 k-mers: {sum(neg_motifs[32])}/{len(neg_sentences)} ({100*sum(neg_motifs[32])/len(neg_sentences):.1f}%)")
    print(f"  First 64 k-mers: {sum(neg_motifs[64])}/{len(neg_sentences)} ({100*sum(neg_motifs[64])/len(neg_sentences):.1f}%)")
    print(f"  First 128 k-mers: {sum(neg_motifs[128])}/{len(neg_sentences)} ({100*sum(neg_motifs[128])/len(neg_sentences):.1f}%)")
    print(f"  All k-mers: {sum(neg_motifs['all'])}/{len(neg_sentences)} ({100*sum(neg_motifs['all'])/len(neg_sentences):.1f}%)")
    
    # Baseline classifier using k-mer counts
    print(f"\n" + "="*70)
    print("BASELINE CLASSIFIER (Logistic Regression on k-mer counts)")
    print("="*70)
    
    print("\nConverting sequences to k-mer count vectors...")
    X_train = sentences_to_kmer_counts(train_sentences, vocab)
    X_val = sentences_to_kmer_counts(val_sentences, vocab)
    X_test = sentences_to_kmer_counts(test_sentences, vocab)
    
    print(f"Feature matrix shape: {X_train.shape}")
    
    print("\nTraining logistic regression...")
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(X_train, train_labels)
    
    # Evaluate
    train_pred = clf.predict(X_train)
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)
    
    print(f"\nResults:")
    print(f"Train Accuracy: {accuracy_score(train_labels, train_pred):.4f}")
    print(f"Val Accuracy: {accuracy_score(val_labels, val_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(test_labels, test_pred):.4f}")
    
    print(f"\nTest Set Classification Report:")
    print(classification_report(test_labels, test_pred, target_names=['non-promoter', 'promoter']))
    
    # Check most important features
    print(f"\nTop 10 features for promoter class:")
    reverse_vocab = {v: k for k, v in vocab.items()}
    coef = clf.coef_[0]
    top_indices = np.argsort(coef)[-10:][::-1]
    for idx in top_indices:
        kmer = reverse_vocab.get(idx, f'idx_{idx}')
        print(f"  {kmer}: {coef[idx]:.4f}")
    
    print(f"\nTop 10 features for non-promoter class:")
    bottom_indices = np.argsort(coef)[:10]
    for idx in bottom_indices:
        kmer = reverse_vocab.get(idx, f'idx_{idx}')
        print(f"  {kmer}: {coef[idx]:.4f}")
    
    print(f"\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
