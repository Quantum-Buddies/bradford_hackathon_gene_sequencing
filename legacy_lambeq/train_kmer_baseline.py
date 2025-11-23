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
    def __init__(self, num_samples=10000, seq_len=128, k=4):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.k = k
        self.vocab_size = 4**k  # 256 tokens for k=4
        
        # Generate synthetic k-mer indices
        # Real data would parse FASTA -> k-mers -> indices
        # Here we simulate indices directly in range [0, 255]
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
# 3. Training Loop
# ==============================================================================
def train_kmer_baseline():
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
        'epochs': 10,
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
    
    # Dataset with K-mer preprocessing simulation
    dataset = GenomicKmerDataset(num_samples=5000, k=k)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    history = {'train_loss': [], 'train_acc': []}
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            loss = criterion(output.reshape(-1, config['vocab_size']), target.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = output.argmax(dim=-1)
            correct += (preds == target).sum().item()
            total += target.numel()
        
        avg_loss = total_loss / len(dataloader)
        acc = 100 * correct / total
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(acc)
        
        print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
        
    print(f"\nTraining finished in {time.time() - start_time:.2f}s")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, "classical_kmer_baseline.pt")
    
    return history

# ==============================================================================
# 4. Plotting
# ==============================================================================
def plot_kmer_comparison(classical_history):
    epochs = range(1, len(classical_history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, classical_history['train_loss'], 'b--', label='Classical (k=4)')
    # Placeholder for Quixer curve if needed
    plt.title('Loss Comparison (4-mer Tokenization)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, classical_history['train_acc'], 'b--', label='Classical (k=4)')
    plt.title('Accuracy Comparison (4-mer Tokenization)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("kmer_baseline_results.png")
    print("Saved kmer_baseline_results.png")

if __name__ == "__main__":
    hist = train_kmer_baseline()
    plot_kmer_comparison(hist)

