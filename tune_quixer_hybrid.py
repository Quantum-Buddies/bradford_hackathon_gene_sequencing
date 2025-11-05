#!/usr/bin/env python3
"""
Optuna Hyperparameter Tuning for Quixer Hybrid Model
=====================================================
Automatically finds optimal hyperparameters for the Quixer quantum transformer
trained on quantized lambeq embeddings.

Features:
- Dynamic search space for quantum architecture (qubits, layers, ansatz)
- Pruning to terminate unpromising trials early
- Parallel trial execution support
- Visualization of optimization history
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from tqdm import tqdm

# Add Quixer to path
sys.path.insert(0, '/scratch/cbjp404/bradford_hackathon_2025/Quixer')
from quixer.quixer_classifier import QuixerClassifier


# Global variables for cached data
TRAIN_SEQS = None
TRAIN_LABELS = None
VAL_SEQS = None
VAL_LABELS = None
METADATA = None
DEVICE = None
DATA_DIR = None
CLUSTER_CENTERS = None
EPOCHS_PER_TRIAL = 10


def load_quantized_data(data_dir: Path, split: str):
    """Load quantized token sequences."""
    file_path = data_dir / f"{split}.pt"
    data = torch.load(file_path)
    return data['sequences'], data['labels']


def cache_data(data_dir: Path):
    """Load and cache all data once."""
    global TRAIN_SEQS, TRAIN_LABELS, VAL_SEQS, VAL_LABELS, METADATA, CLUSTER_CENTERS
    
    print("Loading and caching data...")
    TRAIN_SEQS, TRAIN_LABELS = load_quantized_data(data_dir, 'train')
    VAL_SEQS, VAL_LABELS = load_quantized_data(data_dir, 'val')
    
    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        METADATA = json.load(f)
    
    print(f"  Train: {len(TRAIN_SEQS)} samples")
    print(f"  Val: {len(VAL_SEQS)} samples")
    print(f"  Vocab size: {METADATA['vocabulary_size']}")
    print(f"  Seq length: {METADATA['seq_len']}")

    cluster_file = data_dir / 'cluster_centers.pt'
    if cluster_file.exists():
        payload = torch.load(cluster_file)
        CLUSTER_CENTERS = payload['centroids'].float()
        print(f"  Loaded cluster centroids: {CLUSTER_CENTERS.shape}")
    else:
        CLUSTER_CENTERS = None
        print("  Warning: cluster_centers.pt not found – embeddings will use random init")


def create_dataloaders(batch_size: int):
    """Create train and validation dataloaders from cached data."""
    train_dataset = TensorDataset(TRAIN_SEQS, TRAIN_LABELS)
    val_dataset = TensorDataset(VAL_SEQS, VAL_LABELS)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid issues with multiprocessing and quantum circuits
        pin_memory=False,  # Disable to prevent deadlocks
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, criterion, trial_num=None, epoch_num=None):
    """Train for one epoch with nested progress bars."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Outer description with trial/epoch info
    desc = "Training"
    if trial_num is not None and epoch_num is not None:
        desc = f"Trial {trial_num} | Epoch {epoch_num} | Training"
    
    pbar = tqdm(train_loader, desc=desc, position=1, leave=False)
    for batch_idx, (sequences, labels) in enumerate(pbar):
        try:
            sequences, labels = sequences.to(DEVICE, non_blocking=False), labels.to(DEVICE, non_blocking=False)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update postfix with metrics
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.1f}%'
            })
            
            # Clear CUDA cache periodically
            if DEVICE.type == 'cuda' and batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            # Skip problematic batches
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
            continue
    
    pbar.close()
    avg_loss = total_loss / max(len(train_loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    
    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, trial_num=None, epoch_num=None):
    """Evaluate model with nested progress bars."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Outer description with trial/epoch info
    desc = "Validation"
    if trial_num is not None and epoch_num is not None:
        desc = f"Trial {trial_num} | Epoch {epoch_num} | Validation"
    
    pbar = tqdm(val_loader, desc=desc, position=1, leave=False)
    with torch.no_grad():
        for sequences, labels in pbar:
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update postfix with metrics
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.1f}%'
            })
    
    pbar.close()
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def objective(trial):
    """
    Optuna objective function.
    
    Returns validation accuracy to be maximized.
    """
    # Sample hyperparameters
    embedding_dim = METADATA['embedding_dim']

    params = {
        'qubits': trial.suggest_int('qubits', 4, 8),
        'layers': trial.suggest_int('layers', 2, 5),
        'ansatz_layers': trial.suggest_int('ansatz_layers', 2, 6),
        'embedding_dim': embedding_dim,
        'dropout': trial.suggest_float('dropout', 0.05, 0.4),
        'lr': trial.suggest_float('lr', 5e-4, 5e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 48, 64]),
    }
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(params['batch_size'])
    
    # Build model
    try:
        model = QuixerClassifier(
            n_qubits=params['qubits'],
            n_tokens=METADATA['seq_len'],
            qsvt_polynomial_degree=params['layers'],
            n_ansatz_layers=params['ansatz_layers'],
            vocabulary_size=METADATA['vocabulary_size'],
            n_classes=METADATA['n_classes'],
            embedding_dimension=params['embedding_dim'],
            dropout=params['dropout'],
            batch_size=params['batch_size'],
            device=DEVICE,
        ).to(DEVICE)
    except Exception as e:
        print(f"Trial {trial.number} failed to build model: {e}")
        raise optuna.TrialPruned()

    if CLUSTER_CENTERS is not None:
        embedding_weight = model.quixer.embedding.weight
        centroids = CLUSTER_CENTERS.to(embedding_weight.device)
        if centroids.shape != embedding_weight.shape:
            raise RuntimeError(
                f"Cluster centroids shape {centroids.shape} does not match embedding weight {embedding_weight.shape}"
            )
        with torch.no_grad():
            embedding_weight.copy_(centroids)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )
    
    # Training loop with pruning (outer epoch bar)
    best_val_acc = 0.0
    
    epoch_bar = tqdm(range(EPOCHS_PER_TRIAL), desc=f"Trial {trial.number}", position=0, leave=True)
    for epoch in epoch_bar:
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, 
                                            trial_num=trial.number, epoch_num=epoch+1)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion,
                                     trial_num=trial.number, epoch_num=epoch+1)
        
        # Update epoch bar postfix with metrics
        epoch_bar.set_postfix({
            'train_acc': f'{train_acc:.1f}%',
            'val_acc': f'{val_acc:.1f}%',
            'best': f'{best_val_acc:.1f}%'
        })
        
        # Report intermediate value
        trial.report(val_acc, epoch)
        
        # Check if should prune
        if trial.should_prune():
            epoch_bar.close()
            raise optuna.TrialPruned()
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    epoch_bar.close()
    return best_val_acc


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for Quixer hybrid model"
    )
    
    parser.add_argument('--data_dir', type=str,
                        default='/scratch/cbjp404/bradford_hackathon_2025/quantized_embeddings',
                        help='Directory containing quantized embeddings')
    parser.add_argument('--output_dir', type=str,
                        default='/scratch/cbjp404/bradford_hackathon_2025/optuna_results',
                        help='Output directory for results')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of optimization trials')
    parser.add_argument('--epochs_per_trial', type=int, default=10,
                        help='Training epochs per trial')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--study_name', type=str, default='quixer_hybrid_tuning',
                        help='Optuna study name')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs (1 for sequential)')
    
    args = parser.parse_args()
    
    # Setup
    global DEVICE, DATA_DIR, EPOCHS_PER_TRIAL
    DEVICE = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    DATA_DIR = Path(args.data_dir)
    EPOCHS_PER_TRIAL = args.epochs_per_trial
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("="*70)
    print("OPTUNA HYPERPARAMETER TUNING FOR QUIXER HYBRID")
    print("="*70)
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Trials: {args.n_trials}")
    print(f"Epochs per trial: {args.epochs_per_trial}")
    print(f"Parallel jobs: {args.n_jobs}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Cache data once
    cache_data(DATA_DIR)
    print()
    
    # Create Optuna study
    print("Creating Optuna study...")
    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',  # Maximize validation accuracy
        sampler=TPESampler(seed=args.seed),  # Tree-structured Parzen Estimator
        pruner=MedianPruner(
            n_startup_trials=5,  # Don't prune first 5 trials
            n_warmup_steps=3,    # Wait 3 epochs before pruning
            interval_steps=1     # Check every epoch
        )
    )
    
    print(f"Study name: {study.study_name}")
    print(f"Sampler: TPESampler")
    print(f"Pruner: MedianPruner")
    print()
    
    # Run optimization
    print("="*70)
    print("STARTING HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    print()
    
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True
    )
    
    # Results
    print()
    print("="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print()
    
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Trial number: {best_trial.number}")
    print(f"  Validation accuracy: {best_trial.value:.2f}%")
    print(f"  Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    print()
    
    # Statistics
    print("Optimization statistics:")
    print(f"  Total trials: {len(study.trials)}")
    print(f"  Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"  Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    print()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save best parameters
    best_params_file = output_dir / f'best_params_{timestamp}.json'
    # Ensure fixed parameters from metadata are preserved in saved config
    best_params_payload = {
        'best_trial_number': best_trial.number,
        'best_value': best_trial.value,
        'best_params': {
            **best_trial.params,
            'embedding_dim': METADATA['embedding_dim'],
        },
        'study_name': study.study_name,
        'n_trials': len(study.trials),
        'timestamp': timestamp
    }

    with open(best_params_file, 'w') as f:
        json.dump(best_params_payload, f, indent=2)
    print(f"Best parameters saved to: {best_params_file}")
    
    # Save trials dataframe
    df = study.trials_dataframe()
    df_file = output_dir / f'trials_{timestamp}.csv'
    df.to_csv(df_file, index=False)
    print(f"All trials saved to: {df_file}")
    
    # Create visualizations
    try:
        import optuna.visualization as vis
        
        print("\nGenerating visualizations...")
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(str(output_dir / f'optimization_history_{timestamp}.html'))
        print(f"  Optimization history: optimization_history_{timestamp}.html")
        
        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(str(output_dir / f'parallel_coordinate_{timestamp}.html'))
        print(f"  Parallel coordinate: parallel_coordinate_{timestamp}.html")
        
        # Parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html(str(output_dir / f'param_importances_{timestamp}.html'))
        print(f"  Parameter importances: param_importances_{timestamp}.html")
        
        # Slice plot
        fig = vis.plot_slice(study)
        fig.write_html(str(output_dir / f'slice_{timestamp}.html'))
        print(f"  Slice plot: slice_{timestamp}.html")
        
    except ImportError:
        print("\nSkipping visualizations (install plotly for visualizations)")
    
    print()
    print("="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\nTrain final model with best hyperparameters:")
    print(f"\nCUDA_VISIBLE_DEVICES=0 python train_quixer_hybrid.py \\")
    print(f"  --data_dir {args.data_dir} \\")
    best_embedding_dim = METADATA['embedding_dim']
    print(f"  --qubits {best_trial.params['qubits']} \\")
    print(f"  --layers {best_trial.params['layers']} \\")
    print(f"  --ansatz_layers {best_trial.params['ansatz_layers']} \\")
    print(f"  --embedding_dim {best_embedding_dim} \\")
    print(f"  --dropout {best_trial.params['dropout']:.4f} \\")
    print(f"  --batch_size {best_trial.params['batch_size']} \\")
    print(f"  --lr {best_trial.params['lr']:.6f} \\")
    print(f"  --weight_decay {best_trial.params['weight_decay']:.6f} \\")
    print(f"  --epochs 50 \\")
    print(f"  --scheduler cosine")
    print()
    print("="*70)


if __name__ == "__main__":
    main()
