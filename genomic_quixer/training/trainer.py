import time
import math
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def train_epoch(model, data_loader, optimizer, criterion, clip, device, vocab_size=None):
    """Train for one epoch."""
    model.train()
    
    epoch_loss = 0
    running_correct = 0
    running_total = 0

    pbar = tqdm(data_loader, desc="Training", dynamic_ncols=True)
    for i, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Handle different model outputs
        # Quixer returns (logits, final_probabilities)
        # ClassicalTransformer returns logits
        output = model(x)
        if isinstance(output, tuple):
            yhat = output[0]
        else:
            yhat = output
            
        # Check shapes for loss calculation
        if yhat.dim() > 2 and yhat.size(1) != vocab_size:
             # For sequence-to-sequence models (like transformer baseline might be if not careful), reshape
             # But here we expect [batch, vocab_size] for next-token prediction
             pass

        loss = criterion(yhat, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        preds = torch.argmax(yhat, dim=1)

        # Update running accuracy for interactive feedback
        correct = (preds == y).sum().item()
        running_correct += correct
        running_total += y.size(0)

        running_loss = epoch_loss / (i + 1)
        running_acc = running_correct / max(running_total, 1)
        pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc*100:.2f}%")
    
    accuracy = running_correct / running_total
    
    return epoch_loss / len(data_loader), accuracy


def evaluate(model, data_loader, criterion, device):
    """Evaluate model."""
    model.eval()
    
    epoch_loss = 0
    running_correct = 0
    running_total = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating", dynamic_ncols=True)
        for i, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            
            output = model(x)
            if isinstance(output, tuple):
                yhat = output[0]
            else:
                yhat = output
                
            loss = criterion(yhat, y)
            epoch_loss += loss.item()
            
            preds = torch.argmax(yhat, dim=1)
            
            # Update running accuracy for interactive feedback
            correct = (preds == y).sum().item()
            running_correct += correct
            running_total += y.size(0)

            running_loss = epoch_loss / (i + 1)
            running_acc = running_correct / max(running_total, 1)
            pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc*100:.2f}%")
    
    accuracy = running_correct / running_total
    avg_loss = epoch_loss / len(data_loader)
    perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
    
    return avg_loss, accuracy, perplexity


def plot_metrics(train_losses, val_losses, train_accs, val_accs, val_ppls, save_path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(train_losses, label='Train')
    axes[0].plot(val_losses, label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss vs Epoch')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(train_accs, label='Train')
    axes[1].plot(val_accs, label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Next-Base Accuracy vs Epoch')
    axes[1].legend()
    axes[1].grid(True)
    
    # Perplexity
    axes[2].plot(val_ppls, label='Val', color='orange')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Perplexity')
    axes[2].set_title('Perplexity vs Epoch (lower is better)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")
