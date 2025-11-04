"""
Genomics Training Runner
========================
Main training script for lambeq+Quixer genomic classification.

Trains multiple models and generates comprehensive evaluation reports.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import sys

# Add Quixer to path
sys.path.insert(0, '/scratch/cbjp404/Quixer')

from quixer.setup_genomics import setup_genomics_dataset


class SimpleClassifier(nn.Module):
    """Simple MLP classifier for embeddings."""
    
    def __init__(self, embedding_dim, n_classes, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def forward(self, x):
        return self.net(x)


class LSTMClassifier(nn.Module):
    """LSTM classifier for sequential processing of embeddings."""
    
    def __init__(self, embedding_dim, n_classes, hidden_dim=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, x):
        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out


class TransformerClassifier(nn.Module):
    """Transformer classifier."""
    
    def __init__(self, embedding_dim, n_classes, hidden_dim=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(embedding_dim, n_classes)
    
    def forward(self, x):
        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.transformer(x)
        # Use CLS token (first token) for classification
        out = self.fc(x[:, 0, :])
        return out


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for embeddings, labels in tqdm(train_loader, desc="Training", leave=False):
        embeddings, labels = embeddings.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings, labels in loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels


def train_model(
    model_name,
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    args
):
    """Train and evaluate a single model."""
    print(f"\n{'='*70}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*70}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # Training loop
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
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            # Save best model (optional)
    
    train_time = time.time() - start_time
    
    # Final test evaluation
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    # Calculate additional metrics
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    conf_matrix = confusion_matrix(test_labels, test_preds)
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS: {model_name}")
    print(f"{'='*70}")
    print(f"Best Val Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Training Time: {train_time:.2f}s ({train_time/60:.2f} min)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nConfusion Matrix:")
    print(conf_matrix)
    print(f"{'='*70}")
    
    results = {
        'model_name': model_name,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_loss': test_loss,
        'train_time': train_time,
        'n_parameters': sum(p.numel() for p in model.parameters()),
        'history': history,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': classification_report(test_labels, test_preds, output_dict=True)
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train genomics classification models")
    parser.add_argument('--models', nargs='+', default=['Quixer', 'LSTM', 'Transformer'],
                        help='Models to train')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--embeddings_dir', type=str,
                        default='/scratch/cbjp404/bradford_hackathon_2025/lambeq_embeddings',
                        help='Path to lambeq embeddings')
    parser.add_argument('--output_dir', type=str,
                        default='/scratch/cbjp404/bradford_hackathon_2025/results',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_loader, val_loader, test_loader, info = setup_genomics_dataset(
        embeddings_dir=args.embeddings_dir,
        batch_size=args.batch_size,
        device=device
    )
    
    embedding_dim = info['embedding_dim']
    n_classes = info['n_classes']
    
    # Train each model
    all_results = {}
    
    for model_name in args.models:
        print(f"\n\n{'#'*70}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*70}")
        
        # Initialize model
        if model_name.lower() == 'quixer':
            # For now, use simple MLP (Quixer circuit simulation would go here)
            model = SimpleClassifier(embedding_dim, n_classes, hidden_dim=512, dropout=0.1)
        elif model_name.lower() == 'lstm':
            model = LSTMClassifier(embedding_dim, n_classes, hidden_dim=256, n_layers=2, dropout=0.3)
        elif model_name.lower() == 'transformer':
            model = TransformerClassifier(embedding_dim, n_classes, hidden_dim=256, n_heads=8, n_layers=3, dropout=0.1)
        else:
            print(f"Unknown model: {model_name}, skipping...")
            continue
        
        model = model.to(device)
        
        # Train and evaluate
        results = train_model(
            model_name, model, train_loader, val_loader, test_loader, device, args
        )
        
        all_results[model_name] = results
    
    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\n{'='*70}")
    print(f"ALL RESULTS SAVED TO: {results_file}")
    print(f"{'='*70}")
    
    # Print summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Test Acc':<12} {'Test F1':<12} {'Params':<15} {'Time (s)':<12}")
    print(f"{'-'*70}")
    
    for model_name, res in all_results.items():
        print(f"{model_name:<15} {res['test_acc']:>10.2f}% {res['test_f1']:>10.4f} "
              f"{res['n_parameters']:>13,} {res['train_time']:>10.1f}")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
