import argparse
import json
import time
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# Simplified imports due to package __init__s
from genomic_quixer.data import GenomicPredictionDataset
from genomic_quixer.models import Quixer, ClassicalTransformer
from genomic_quixer.training import train_epoch, evaluate, plot_metrics

def main():
    parser = argparse.ArgumentParser(description="Train Genomic Quixer or Classical Baseline")
    parser.add_argument('--model_type', type=str, default='quixer', choices=['quixer', 'classical'], help='Model type to train')
    parser.add_argument('--data_dir', type=str, default='genomic_data_preprocessed_kmers', help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    
    # Common hyperparameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--window_size', type=int, default=128, help='Context window size')
    parser.add_argument('--stride', type=int, default=64, help='Stride for data loading')
    parser.add_argument('--embedding_dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Quixer specific
    parser.add_argument('--qubits', type=int, default=6, help='Number of qubits')
    parser.add_argument('--layers', type=int, default=4, help='QSVT polynomial degree')
    parser.add_argument('--ansatz_layers', type=int, default=5, help='Ansatz layers')
    
    # Classical specific
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    data_dir = Path(args.data_dir)
    metadata_path = data_dir / 'metadata.json'
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}. Please run preprocess.py first.")
        
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    vocab_size = metadata['vocab_size']
    
    print(f"\nDataset Metadata:")
    print(f"  Vocab Size: {vocab_size}")
    print(f"  K-mer: {metadata.get('k_mer', 'N/A')}")
    
    # Create Datasets
    print("\n=== Loading Datasets ===")
    train_dataset = GenomicPredictionDataset(data_dir=data_dir, window_size=args.window_size, stride=args.stride, split='train', seed=args.seed)
    val_dataset = GenomicPredictionDataset(data_dir=data_dir, window_size=args.window_size, stride=args.stride, split='val', seed=args.seed)
    
    # Use smaller batch size for validation/inference to avoid OOM if needed, but here double is standard for inference
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size*2, shuffle=False, num_workers=8, pin_memory=True)
    
    # Initialize Model
    print(f"\n=== Initializing {args.model_type.upper()} Model ===")
    if args.model_type == 'quixer':
        model = Quixer(
            n_qubits=args.qubits,
            n_tokens=args.window_size,
            qsvt_polynomial_degree=args.layers,
            n_ansatz_layers=args.ansatz_layers,
            vocabulary_size=vocab_size,
            embedding_dimension=args.embedding_dim,
            dropout=0.15,
            batch_size=args.batch_size,
            device=device
        )
    else:
        model = ClassicalTransformer(
            vocab_size=vocab_size,
            d_model=args.embedding_dim,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.embedding_dim * 4,
            max_len=args.window_size + 1 
        )
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {n_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training Loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    val_ppls = []
    best_val_loss = float('inf')
    
    timestamp = int(time.time())
    checkpoint_path = output_dir / f"{args.model_type}_model_{timestamp}.pt"
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    try:
        for epoch in range(args.epochs):
            start_time = time.time()
            
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, 1.0, device, vocab_size)
            val_loss, val_acc, val_ppl = evaluate(model, val_loader, criterion, device)
            
            elapsed = time.time() - start_time
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            val_ppls.append(val_ppl)
            
            print(f"Epoch {epoch+1}/{args.epochs} | Time: {elapsed:.1f}s")
            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
            print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | PPL: {val_ppl:.2f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': vars(args),
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"  -> Saved best model to {checkpoint_path}")
            print()
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")
    
    # Plotting
    plot_path = output_dir / f"{args.model_type}_training_curves_{timestamp}.png"
    plot_metrics(train_losses, val_losses, train_accs, val_accs, val_ppls, plot_path)
    
    print("Done!")

if __name__ == "__main__":
    main()
