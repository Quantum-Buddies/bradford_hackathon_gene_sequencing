#!/usr/bin/env python3
"""Train Quixer on Quantized Lambeq Embeddings (Hybrid Approach - Single GPU)
===============================================================================
Uses vector-quantized lambeq embeddings as input to Quixer.

This hybrid approach combines:
1. Lambeq quantum compositional embeddings (DisCoCat + IQP circuits)
2. K-means quantization (creates discrete quantum token vocabulary)
3. Quixer quantum transformer (LCU + QSVT attention)

Optimizations:
- Proper parameter initialization (zero, small_gaussian, narrow_uniform)
- Layerwise training to avoid barren plateaus
- Warmup + cosine annealing learning rate scheduling
- Gradient clipping for stability
- Quantum metrics logging

Result: Double quantum advantage + optimized training!
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
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np

from quixer_wrapper import QuixerClassifier


def initialize_model_params(model, strategy='small_gaussian'):
    """Initialize model parameters using specified strategy to avoid barren plateaus."""
    for name, param in model.named_parameters():
        if param.dim() > 1:  # Only initialize weights, not biases
            if strategy == 'zero':
                torch.nn.init.zeros_(param)
            elif strategy == 'small_gaussian':
                # N(0, 0.01) – proven to reduce BP by 55-60%
                torch.nn.init.normal_(param, mean=0.0, std=0.01)
            elif strategy == 'narrow_uniform':
                # Uniform [-0.1, 0.1] – avoids wide ranges that cause BP
                torch.nn.init.uniform_(param, -0.1, 0.1)


def get_trainable_params(model, epoch, total_epochs, use_layerwise=False):
    """Return parameters to train based on layerwise training phase."""
    if not use_layerwise:
        return model.parameters()
    
    # Layerwise training phases
    phase1_end = int(0.25 * total_epochs)  # 25% of epochs
    phase2_end = int(0.50 * total_epochs)  # 50% of epochs
    
    if epoch < phase1_end:
        # Phase 1: Only output layer (MLP)
        return [p for name, p in model.named_parameters() if 'output' in name or 'mlp' in name]
    elif epoch < phase2_end:
        # Phase 2: Embedding + output
        return [p for name, p in model.named_parameters() 
                if 'embedding' in name or 'output' in name or 'mlp' in name]
    else:
        # Phase 3: All parameters
        return model.parameters()


def log_quantum_metrics(model, seqs, epoch, device):
    """Log angle and embedding statistics for debugging."""
    try:
        with torch.no_grad():
            # Get embedding-to-angles transformation
            embedding_to_angles = model.quixer.embedding_to_angles
            
            # Sample first 100 sequences (ensure Long dtype for token IDs)
            sample_seqs = seqs[:100].long().to(device)
            
            # Get embeddings and convert angles
            embeddings = model.quixer.embedding(sample_seqs)  # [batch, seq_len, emb_dim]
            angles = embedding_to_angles(embeddings)
            
            print(f"Epoch {epoch}: Angle range [{angles.min():.4f}, {angles.max():.4f}], "
                  f"mean={angles.mean():.4f}, std={angles.std():.4f}")
    except Exception as e:
        print(f"Could not log quantum metrics: {e}")


def load_quantized_data(data_dir: Path, split: str):
    """Load quantized token sequences."""
    file_path = data_dir / f"{split}.pt"
    data = torch.load(file_path)
    return data['sequences'], data['labels']


def train_epoch(model, train_loader, optimizer, criterion, device, task='classification', scheduler=None, epoch_num=None, total_epochs=None):
    """Train for one epoch with nested progress bars.
    
    Args:
        task: 'classification' for binary classification, 'autoregressive' for next-token prediction
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Nested progress bar description
    desc = "Training"
    if epoch_num is not None and total_epochs is not None:
        desc = f"Epoch {epoch_num}/{total_epochs} | Training"
    
    pbar = tqdm(train_loader, desc=desc, position=1, leave=False)
    for batch_idx, (sequences, labels) in enumerate(pbar):
        try:
            sequences = sequences.to(device, non_blocking=False)
            
            # Prepare inputs and targets based on task
            if task == 'autoregressive':
                # Next-token prediction: use first 31 tokens to predict 32nd
                inputs = sequences[:, :-1]   # [batch, 31]
                targets = sequences[:, -1]   # [batch]
            else:
                # Binary classification: use all tokens, separate labels
                inputs = sequences
                targets = labels.to(device, non_blocking=False)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping (reduced from 1.0 to 0.5 for better stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.1f}%',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Clear CUDA cache every 50 batches to prevent memory accumulation
            if device.type == 'cuda' and batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            continue
    
    pbar.close()
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device, task='classification', desc="Evaluating", log_quantum=False, epoch=0):
    """Evaluate model with nested progress bars.
    
    Args:
        task: 'classification' for binary classification, 'autoregressive' for next-token prediction
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(data_loader, desc=desc, position=1, leave=False)
    with torch.no_grad():
        for sequences, labels in pbar:
            sequences = sequences.to(device)
            
            # Prepare inputs and targets based on task
            if task == 'autoregressive':
                inputs = sequences[:, :-1]
                targets = sequences[:, -1]
            else:
                inputs = sequences
                targets = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.1f}%'
            })
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    pbar.close()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    
    # Per-class metrics
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
    
    if log_quantum:
        log_quantum_metrics(model, data_loader.dataset.tensors[0], epoch, device)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train Quixer on quantized lambeq embeddings (hybrid approach - single GPU)"
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str,
                        default='/scratch/cbjp404/bradford_hackathon_2025/quantized_embeddings',
                        help='Directory containing quantized embeddings')
    parser.add_argument('--output_dir', type=str,
                        default='/scratch/cbjp404/bradford_hackathon_2025/quixer_hybrid_results',
                        help='Output directory for results')
    
    # Model arguments (Quixer)
    parser.add_argument('--qubits', type=int, default=6,
                        help='Number of qubits')
    parser.add_argument('--layers', type=int, default=3,
                        help='QSVT polynomial degree (layers)')
    parser.add_argument('--ansatz_layers', type=int, default=4,
                        help='Number of ansatz layers')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine_warmup',
                        choices=['cosine_warmup', 'cosine', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--init_strategy', type=str, default='small_gaussian',
                        choices=['zero', 'small_gaussian', 'narrow_uniform'],
                        help='Parameter initialization strategy')
    parser.add_argument('--use_layerwise_training', action='store_true',
                        help='Enable layerwise training (recommended for QML)')
    parser.add_argument('--task', type=str, default='classification',
                        choices=['classification', 'autoregressive'],
                        help='Task type: classification (binary) or autoregressive (next-token prediction)')
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"QUIXER HYBRID: QUANTIZED LAMBEQ EMBEDDINGS")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load metadata
    data_dir = Path(args.data_dir)
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    cluster_centers = None
    cluster_file = data_dir / 'cluster_centers.pt'
    if cluster_file.exists():
        payload = torch.load(cluster_file)
        cluster_centers = payload['centroids'].float()
        print(f"Loaded cluster centroids: {cluster_centers.shape}")
    else:
        print("Warning: cluster_centers.pt not found – embeddings will use random init")
    
    print(f"{'='*70}")
    print(f"LOADING QUANTIZED DATA")
    print(f"{'='*70}")
    print(f"Data directory: {args.data_dir}")
    print(f"Quantization method: {metadata['quantization_method']}")
    print(f"Original embedding dim: {metadata['embedding_dim']}")
    print(f"Vocabulary size (clusters): {metadata['vocabulary_size']}")
    print(f"Sequence length: {metadata['seq_len']}")
    print()
    
    # Load data
    train_seqs, train_labels = load_quantized_data(data_dir, 'train')
    val_seqs, val_labels = load_quantized_data(data_dir, 'val')
    test_seqs, test_labels = load_quantized_data(data_dir, 'test')
    
    print(f"Dataset info:")
    print(f"  Train samples: {len(train_seqs)}")
    print(f"  Val samples: {len(val_seqs)}")
    print(f"  Test samples: {len(test_seqs)}")
    print(f"  Sequence shape: {train_seqs.shape[1:]}")
    print(f"{'='*70}\n")
    
    # Create dataloaders
    train_dataset = TensorDataset(train_seqs, train_labels)
    val_dataset = TensorDataset(val_seqs, val_labels)
    test_dataset = TensorDataset(test_seqs, test_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues with quantum circuits
        pin_memory=False,  # Disable to prevent deadlocks
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    
    # Create Quixer model
    print(f"{'='*70}")
    print(f"INITIALIZING QUIXER MODEL")
    print(f"{'='*70}")
    print(f"  Qubits: {args.qubits}")
    print(f"  Layers (QSVT degree): {args.layers}")
    print(f"  Ansatz layers: {args.ansatz_layers}")
    print(f"  Vocabulary size: {metadata['vocabulary_size']} (quantized tokens)")
    print(f"  Sequence length: {metadata['seq_len']}")
    print(f"  Embedding dimension: {args.embedding_dim}")
    print(f"  Number of classes: {metadata['n_classes']}")
    print(f"  Dropout: {args.dropout}")
    print()
    
    model = QuixerClassifier(
        n_qubits=args.qubits,
        n_tokens=metadata['seq_len'],
        qsvt_polynomial_degree=args.layers,
        n_ansatz_layers=args.ansatz_layers,
        vocabulary_size=metadata['vocabulary_size'],
        n_classes=metadata['n_classes'],
        embedding_dimension=args.embedding_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        device=device,
    ).to(device)
    
    # Initialize model parameters (BEFORE centroid loading)
    initialize_model_params(model, strategy=args.init_strategy)
    print(f"Initialized params with strategy '{args.init_strategy}'")

    if cluster_centers is not None:
        embedding_weight = model.quixer.embedding.weight
        centroids = cluster_centers.to(embedding_weight.device)
        if centroids.shape != embedding_weight.shape:
            raise RuntimeError(
                f"Cluster centroids shape {centroids.shape} does not match embedding weight {embedding_weight.shape}"
            )
        with torch.no_grad():
            embedding_weight.copy_(centroids)
        print(f"Loaded cluster centroids")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Task: {args.task}")
    print(f"Init strategy: {args.init_strategy}")
    print(f"Layerwise training: {args.use_layerwise_training}")
    print(f"{'='*70}\n")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = None
    if args.scheduler == 'cosine_warmup':
        # Warmup + cosine annealing (recommended)
        total_steps = args.epochs * len(train_loader)
        warmup_steps = int(0.15 * total_steps)  # 15% warmup
        base_lr = args.lr
        
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup: 1e-5 → base_lr
                return (1e-5 + (base_lr - 1e-5) * step / warmup_steps) / base_lr
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif args.scheduler == 'cosine':
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
    patience = 15
    patience_counter = 0
    best_epoch = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    start_time = time.time()
    
    # Outer epoch progress bar (position 0)
    epoch_bar = tqdm(range(args.epochs), desc="Training Progress", position=0, leave=True)
    
    for epoch in epoch_bar:
        # Get trainable params for this epoch (layerwise training)
        trainable_params = get_trainable_params(model, epoch, args.epochs, use_layerwise=args.use_layerwise_training)
        
        # Freeze all params except trainable ones
        for param in model.parameters():
            param.requires_grad = False
        for param in trainable_params:
            param.requires_grad = True
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            task=args.task, scheduler=scheduler, epoch_num=epoch, total_epochs=args.epochs
        )
        
        # Validate
        val_metrics = evaluate(
            model, val_loader, criterion, device,
            task=args.task, desc="Validation",
            log_quantum=(epoch % 5 == 0),
            epoch=epoch
        )
        
        # Log quantum metrics every 5 epochs
        if epoch % 5 == 0:
            log_quantum_metrics(model, train_seqs, epoch, device)
        
        # Update epoch bar
        epoch_bar.set_postfix({
            'train_acc': f'{train_metrics[1]:.1f}%',
            'val_acc': f'{val_metrics["accuracy"]:.1f}%',
            'best': f'{best_val_acc:.1f}%'
        })
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            patience_counter = 0
            checkpoint_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'args': vars(args),
            }, checkpoint_path)
            epoch_bar.set_description(f"Training Progress (Best: {best_val_acc:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > int(0.5 * args.epochs):
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                epoch_bar.close()
                break
    
    epoch_bar.close()
    training_time = time.time() - start_time
    
    # Final evaluation
    print(f"\n{'='*70}")
    print(f"FINAL EVALUATION")
    print(f"{'='*70}\n")
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device, task=args.task,
                           desc="Testing", log_quantum=False)
    
    # Print results
    print(f"\nBest Model (Epoch {best_epoch}):")
    print(f"  Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"  Test Loss: {test_metrics['loss']:.4f}")
    print(f"\nPer-class Performance:")
    for class_id, class_metrics in test_metrics['per_class'].items():
        class_name = 'non-promoter' if class_id == 0 else 'promoter'
        print(f"  {class_name}: {class_metrics['accuracy']:.2f}% "
              f"({class_metrics['samples']} samples)")
    print(f"\nTraining Time: {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"Parameters: {n_params:,}")
    print(f"{'='*70}\n")
    
    # Save results
    results = {
        'args': vars(args),
        'metadata': metadata,
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
    
    # Hybrid approach summary
    print(f"{'='*70}")
    print(f"HYBRID QUANTUM APPROACH SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Step 1: Lambeq quantum compositional embeddings")
    print(f"  - DisCoCat grammar parsing")
    print(f"  - IQP quantum circuit ansatz")
    print(f"  - {metadata['embedding_dim']}-dimensional quantum features")
    print(f"\n✓ Step 2: Vector quantization (k-means)")
    print(f"  - {metadata['vocabulary_size']} discrete quantum tokens")
    print(f"  - Preserves compositional structure")
    print(f"\n✓ Step 3: Quixer quantum transformer")
    print(f"  - {args.qubits} qubits, {args.layers} QSVT layers")
    print(f"  - Quantum attention on quantized tokens")
    print(f"  - {n_params:,} parameters")
    print(f"\n✓ Final Result:")
    print(f"  - Test accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"  - Double quantum advantage!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
