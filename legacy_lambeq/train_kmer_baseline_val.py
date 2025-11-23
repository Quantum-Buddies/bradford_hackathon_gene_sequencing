import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import itertools
from pathlib import Path

# ==============================================================================
# 1. K-mer Dataset (k=4, Vocab Size=4^4=256)
# ==============================================================================
class GenomicKmerDataset(Dataset):
    def __init__(self, num_samples=10000, seq_len=128, k=4, seed=42):
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.k = k
        self.vocab_size = 4**k  # 256 tokens for k=4
        
        # Generate synthetic k-mer indices
        self.data = torch.randint(0, self.vocab_size, (num_samples, seq_len + 1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]

# ==============================================================================
# 2. Classical Transformer Model (Adapted for K-mer Vocab)
# ==============================================================================
class ClassicalTransformer(nn.Module):
    def __init__(self, vocab_size=256, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, d_model))
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src):
        seq_len = src.size(1)
        src = self.embedding(src) + self.pos_encoder[:, :seq_len, :]
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

# ==============================================================================
# 3. Training Loop with Validation
# ==============================================================================
def train_kmer_baseline_with_validation():
    k = 4
    vocab_size = 4**k  # 256
    print(f"Initializing Classical Transformer Baseline (k={k}, vocab={vocab_size})...")
    
    config = {
        'vocab_size': vocab_size,
        'd_model': 64, 
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 256,
        'batch_size': 32,
        'epochs': 15, # Increased slightly to see divergence
        'lr': 0.001
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ClassicalTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward']
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Classical Model Parameters: {param_count:,}")
    
    # Train/Val Split
    train_dataset = GenomicKmerDataset(num_samples=8000, k=k, seed=42)
    val_dataset = GenomicKmerDataset(num_samples=2000, k=k, seed=999) # Different seed for validation
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        # --- TRAIN LOOP ---
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output.reshape(-1, config['vocab_size']), target.reshape(-1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = output.argmax(dim=-1)
            train_correct += (preds == target).sum().item()
            train_total += target.numel()
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # --- VALIDATION LOOP ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                loss = criterion(output.reshape(-1, config['vocab_size']), target.reshape(-1))
                val_loss += loss.item()
                
                preds = output.argmax(dim=-1)
                val_correct += (preds == target).sum().item()
                val_total += target.numel()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
        
    print(f"\nTraining finished in {time.time() - start_time:.2f}s")
    
    return history

# ==============================================================================
# 4. Plotting
# ==============================================================================
def plot_val_comparison(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r--', label='Validation Loss')
    plt.title('Loss: Train vs Validation (Generalization Check)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], 'r--', label='Validation Accuracy')
    plt.title('Accuracy: Train vs Validation (Generalization Check)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("kmer_baseline_validation.png")
    print("Saved kmer_baseline_validation.png")

if __name__ == "__main__":
    hist = train_kmer_baseline_with_validation()
    plot_val_comparison(hist)

