#!/usr/bin/env python3
"""
Train Quixer on Genomics Classification
========================================
Uses the real Quixer quantum transformer model for genomic sequence classification.
"""

import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Add Quixer to path
sys.path.insert(0, '/scratch/cbjp404/bradford_hackathon_2025/Quixer')

from quixer.genomics_dataset import setup_genomics_kmer_dataset
from quixer.quixer_classifier import QuixerClassifier


def train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (sequences, labels) in enumerate(pbar):
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device, desc="Evaluating"):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(data_loader, desc=desc):
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    
    # Calculate per-class accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'per_class': {}
    }
    
    for class_id in np.unique(all_labels):
        class_mask = all_labels == class_id
        class_correct = (all_preds[class_mask] == all_labels[class_mask]).sum()
        class_total = class_mask.sum()
        class_acc = 100.0 * class_correct / class_total if class_total > 0 else 0.0
        metrics['per_class'][int(class_id)] = {
            'accuracy': class_acc,
            'samples': int(class_total)
        }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Quixer on Genomics Classification")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str,
                        default='/scratch/cbjp404/bradford_hackathon_2025/processed_data',
                        help='Path to processed genomics data')
    parser.add_argument('--output_dir', type=str,
                        default='/scratch/cbjp404/bradford_hackathon_2025/quixer_results',
                        help='Output directory for results')
    
    # Model arguments
    parser.add_argument('--qubits', type=int, default=6,
                        help='Number of qubits')
    parser.add_argument('--layers', type=int, default=3,
                        help='QSVT polynomial degree (layers)')
    parser.add_argument('--ansatz_layers', type=int, default=4,
                        help='Number of ansatz layers')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--max_seq_len', type=int, default=64,
                        help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['none', 'cosine', 'step'],
                        help='Learning rate scheduler')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"QUIXER GENOMICS CLASSIFICATION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    train_loader, val_loader, test_loader, info = setup_genomics_kmer_dataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_len=args.max_seq_len,
        device=device
    )
    
    # Create model
    print(f"{'='*70}")
    print(f"INITIALIZING QUIXER MODEL")
    print(f"{'='*70}")
    print(f"  Qubits: {args.qubits}")
    print(f"  Layers (QSVT degree): {args.layers}")
    print(f"  Ansatz layers: {args.ansatz_layers}")
    print(f"  Vocabulary size: {info['vocab_size']}")
    print(f"  Sequence length: {info['n_tokens']}")
    print(f"  Embedding dimension: {args.embedding_dim}")
    print(f"  Number of classes: {info['n_classes']}")
    print(f"  Dropout: {args.dropout}")
    print()
    
    model = QuixerClassifier(
        n_qubits=args.qubits,
        n_tokens=info['n_tokens'],
        qsvt_polynomial_degree=args.layers,
        n_ansatz_layers=args.ansatz_layers,
        vocabulary_size=info['vocab_size'],
        n_classes=info['n_classes'],
        embedding_dimension=args.embedding_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        device=device,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"{'='*70}\n")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop
    print(f"{'='*70}")
    print(f"TRAINING")
    print(f"{'='*70}\n")
    
    best_val_acc = 0.0
    best_epoch = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, desc="Validation")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            checkpoint_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'args': vars(args),
            }, checkpoint_path)
            print(f"  ✓ New best model saved (Val Acc: {best_val_acc:.2f}%)")
    
    training_time = time.time() - start_time
    
    # Final evaluation on test set
    print(f"\n{'='*70}")
    print(f"FINAL EVALUATION")
    print(f"{'='*70}\n")
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device, desc="Testing")
    
    # Print results
    print(f"\nBest Model (Epoch {best_epoch}):")
    print(f"  Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"  Test Loss: {test_metrics['loss']:.4f}")
    print(f"\nPer-class Performance:")
    for class_id, class_metrics in test_metrics['per_class'].items():
        print(f"  Class {class_id}: {class_metrics['accuracy']:.2f}% "
              f"({class_metrics['samples']} samples)")
    print(f"\nTraining Time: {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"Parameters: {n_params:,}")
    print(f"{'='*70}\n")
    
    # Save results
    results = {
        'args': vars(args),
        'model_params': n_params,
        'training_time': training_time,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'test_metrics': {
            'accuracy': test_metrics['accuracy'],
            'loss': test_metrics['loss'],
            'per_class': test_metrics['per_class']
        },
        'history': history
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f'results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}\n")


if __name__ == "__main__":
    main()
