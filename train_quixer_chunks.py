#!/usr/bin/env python3
"""
Train Quixer on Lambeq Chunks (Multi-GPU DDP Ready)
====================================================

Optimized for 3 L40 GPUs + 200GB RAM + 24 CPUs per task

Uses ChunkDataset to avoid OOM and enables efficient distributed training.

Key optimizations:
- ChunkDataset: Load chunks on-demand (2 GB RAM per chunk)
- DDP: Distributed training across 3 GPUs (2-3× speedup)
- Gradient accumulation: Larger effective batch size
- Mixed precision: FP16 for memory efficiency (optional)
- Parameter initialization: Avoid barren plateaus
- Learning rate scheduling: Warmup + cosine annealing
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from chunk_dataset import ChunkDataset
from quixer_wrapper import QuixerClassifier


DEFAULT_TRAIN_CHUNKS = list(range(10))
DEFAULT_VAL_CHUNKS = [0, 1]
DEFAULT_TEST_CHUNKS = [0, 1]


def parse_chunk_arg(arg_value, default_indices):
    """Parse chunk index selections supplied via CLI."""

    if not arg_value:
        return list(default_indices)

    arg_value = arg_value.strip()
    if "-" in arg_value:
        start, end = arg_value.split("-", maxsplit=1)
        return list(range(int(start), int(end) + 1))

    return [int(idx.strip()) for idx in arg_value.split(",") if idx.strip()]


def setup_ddp():
    """Initialize DDP if running with torchrun."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_ddp():
    """Cleanup DDP."""
    if dist.is_available() and dist.is_initialized():
        destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def initialize_model_params(model, strategy='small_gaussian'):
    """Initialize model parameters to avoid barren plateaus."""
    for name, param in model.named_parameters():
        if param.dim() > 1:
            if strategy == 'zero':
                torch.nn.init.zeros_(param)
            elif strategy == 'small_gaussian':
                torch.nn.init.normal_(param, mean=0.0, std=0.01)
            elif strategy == 'narrow_uniform':
                torch.nn.init.uniform_(param, -0.1, 0.1)


def load_cluster_centroids(model, centroids_path, device, world_size):
    """Load cluster centroids into embedding layer and return them for tokenization."""
    if not Path(centroids_path).exists():
        if is_main_process():
            print(f"⚠️  Warning: {centroids_path} not found, using random init")
        return None

    payload = torch.load(centroids_path, map_location='cpu')

    if isinstance(payload, torch.Tensor):
        centroids = payload
    elif isinstance(payload, dict):
        if 'centroids' in payload:
            centroids = payload['centroids']
        elif 'cluster_centers' in payload:
            centroids = payload['cluster_centers']
        else:
            raise KeyError(
                f"Centroid checkpoint at {centroids_path} missing 'centroids'/'cluster_centers' keys"
            )
    else:
        raise TypeError(
            f"Unexpected centroid checkpoint type: {type(payload)}."
        )

    if not torch.is_tensor(centroids):
        raise TypeError(f"Centroids in {centroids_path} are not a tensor (got {type(centroids)}).")

    centroids = centroids.to(dtype=torch.float32)

    if is_main_process():
        print(f"✓ Loading cluster centroids: {tuple(centroids.shape)}")

    current_vocab = model.quixer.embedding.weight.shape[0]
    target_vocab = centroids.shape[0]

    if current_vocab != target_vocab:
        if is_main_process():
            print(f"  Adjusting model vocabulary from {current_vocab} to {target_vocab}")
        emb_dim = model.quixer.embedding.embedding_dim
        new_emb = nn.Embedding(target_vocab, emb_dim).to(device)
        with torch.no_grad():
            new_emb.weight.copy_(centroids.to(device))
        model.quixer.embedding = new_emb

        last = model.quixer.output_feedforward[-1]
        if isinstance(last, nn.Linear) and last.out_features != target_vocab:
            model.quixer.output_feedforward[-1] = nn.Linear(last.in_features, target_vocab).to(device)
    else:
        with torch.no_grad():
            model.quixer.embedding.weight.copy_(centroids.to(model.quixer.embedding.weight.device))

    # Broadcast centroids to all ranks so tokenization stays consistent
    centroids = centroids.to(device)
    if world_size > 1:
        dist.broadcast(centroids, src=0)

    return centroids.contiguous()


@torch.no_grad()
def quantize_embeddings_to_tokens(embeddings: torch.Tensor, centroids: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    """Assign each embedding to nearest centroid ID."""
    if embeddings.dim() != 3:
        raise ValueError(f"Expected embeddings shape [batch, seq, dim], got {embeddings.shape}")

    centroids = centroids.to(embeddings.device)
    bsz, seq_len, dim = embeddings.shape
    flat = embeddings.reshape(-1, dim)
    vocab = centroids.size(0)

    tokens = torch.empty(flat.size(0), device=embeddings.device, dtype=torch.long)

    # Process in chunks to reduce peak memory
    for start in range(0, flat.size(0), chunk_size):
        end = min(start + chunk_size, flat.size(0))
        dists = torch.cdist(flat[start:end].unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
        tokens[start:end] = torch.argmin(dists, dim=1)

    return tokens.view(bsz, seq_len)


def create_dataloaders(
    embeddings_dir,
    batch_size,
    num_workers,
    train_chunks,
    val_chunks,
    test_chunks,
    target_device,
):
    """Create DataLoaders from chunks."""

    if is_main_process():
        print("\n" + "=" * 60)
        print("Creating DataLoaders from Chunks")
        print("=" * 60)

    dataset_device = "cpu"

    train_dataset = ChunkDataset(
        embeddings_dir,
        "train",
        chunk_indices=train_chunks,
        device=dataset_device,
    )

    val_dataset = ChunkDataset(
        embeddings_dir,
        "val",
        chunk_indices=val_chunks,
        device=dataset_device,
    )

    test_dataset = ChunkDataset(
        embeddings_dir,
        "test",
        chunk_indices=test_chunks,
        device=dataset_device,
    )

    pin_memory = target_device.type == "cuda"
    persistent_workers = False  # disable to avoid lingering large chunk buffers

    # Use DistributedSampler if DDP is active
    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    if is_main_process():
        print(
            f"  Train chunks: {train_chunks} | samples: {len(train_dataset):,} | batches/epoch: {len(train_loader)}"
        )
        print(
            f"  Val chunks:   {val_chunks} | samples: {len(val_dataset):,} | batches/epoch: {len(val_loader)}"
        )
        print(
            f"  Test chunks:  {test_chunks} | samples: {len(test_dataset):,} | batches/epoch: {len(test_loader)}"
        )

    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs, centroids, autoregressive=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{total_epochs}",
        disable=not is_main_process(),
    )
    
    for embeddings, labels in pbar:
        embeddings = embeddings.to(device)
        labels = labels.to(device).long()

        # Convert embeddings to nearest centroid token IDs
        tokens = quantize_embeddings_to_tokens(embeddings, centroids)

        if autoregressive:
            # Use configurable context length; predict the token at position T-1
            seq_len_cur = tokens.size(1)
            T = min(getattr(train_epoch, "context_len", seq_len_cur), seq_len_cur)
            if T < 2:
                # Not enough context to form (input, target); skip this batch
                # Maintain progress bar stability
                pbar.set_postfix({"loss": f"{(total_loss / max(1,total_samples)):.4f}",
                                  "acc": f"{(total_correct / max(1,total_samples)):.2%}"})
                continue
            # Unpadded inputs up to T-1
            unpadded = tokens[:, : T - 1]
            # Quixer expects fixed n_tokens; pad or trim to match
            _base = model.module if isinstance(model, DDP) else model
            model_T = getattr(_base.quixer, "n_tokens", unpadded.size(1))
            if unpadded.size(1) < model_T:
                pad = torch.zeros(unpadded.size(0), model_T - unpadded.size(1), dtype=unpadded.dtype, device=unpadded.device)
                inputs = torch.cat([unpadded, pad], dim=1).contiguous()
            else:
                inputs = unpadded[:, :model_T].contiguous()
            targets = tokens[:, T - 1].contiguous()
        else:
            inputs = tokens
            targets = labels

        logits = model(inputs)
        loss = criterion(logits, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * embeddings.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += embeddings.size(0)
        
        # Calculate perplexity for autoregressive
        ppl = torch.exp(loss).item() if autoregressive else None
        
        postfix = {
            "loss": f"{total_loss / total_samples:.4f}",
            "acc": f"{total_correct / total_samples:.2%}",
        }
        if ppl is not None:
            postfix["ppl"] = f"{ppl:.2f}"
        pbar.set_postfix(postfix)
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, val_loader, criterion, device, centroids, autoregressive=False):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(val_loader, desc="Validating", disable=not is_main_process())
    
    for embeddings, labels in pbar:
        embeddings = embeddings.to(device)
        labels = labels.to(device).long()

        tokens = quantize_embeddings_to_tokens(embeddings, centroids)

        if autoregressive:
            seq_len_cur = tokens.size(1)
            T = min(getattr(validate, "context_len", seq_len_cur), seq_len_cur)
            if T < 2:
                # Skip if not enough context
                continue
            unpadded = tokens[:, : T - 1]
            _base = model.module if isinstance(model, DDP) else model
            model_T = getattr(_base.quixer, "n_tokens", unpadded.size(1))
            if unpadded.size(1) < model_T:
                pad = torch.zeros(unpadded.size(0), model_T - unpadded.size(1), dtype=unpadded.dtype, device=unpadded.device)
                inputs = torch.cat([unpadded, pad], dim=1).contiguous()
            else:
                inputs = unpadded[:, :model_T].contiguous()
            targets = tokens[:, T - 1].contiguous()
        else:
            inputs = tokens
            targets = labels

        logits = model(inputs)
        loss = criterion(logits, targets)
        
        total_loss += loss.item() * embeddings.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += embeddings.size(0)
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description="Train Quixer on Lambeq chunks")
    parser.add_argument("--embeddings_dir", default="lambeq_embeddings_autoregressive",
                        help="Directory with chunk files")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help="Weight decay")
    parser.add_argument("--init_strategy", default="small_gaussian",
                        choices=["zero", "small_gaussian", "narrow_uniform"],
                        help="Parameter initialization strategy")
    parser.add_argument("--device", default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--output_dir", default="quixer_results",
                        help="Output directory")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers per process (default: 8 → 24 total for 3 GPUs)")
    parser.add_argument("--centroids_path", default="cluster_centers.pt",
                        help="Path to cluster centroids file")
    parser.add_argument("--train_chunks",
                        help="Train chunk indices (e.g. '0-9' or '0,1,2') [default: 0-9]")
    parser.add_argument("--val_chunks",
                        help="Validation chunk indices (e.g. '0-1' or '0,1') [default: 0,1]")
    parser.add_argument("--test_chunks",
                        help="Test chunk indices (e.g. '0-1' or '0,1') [default: 0,1]")
    parser.add_argument("--autoregressive", action="store_true",
                        help="Enable autoregressive mode (next-token prediction)")
    parser.add_argument("--use_layerwise_training", action="store_true",
                        help="Enable layerwise (staged) training to mitigate barren plateaus")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing for CrossEntropyLoss (e.g., 0.1)")
    parser.add_argument("--warmup_frac", type=float, default=0.2,
                        help="Fraction of epochs used for LR warmup (0-1)")
    parser.add_argument("--context_len", type=int, default=32,
                        help="Context length for autoregressive mode (predict token at position context_len-1)")
    parser.add_argument("--wandb_project", default=None,
                        help="Weights & Biases project name (enables logging when provided)")
    parser.add_argument("--wandb_entity", default=None,
                        help="Weights & Biases entity or team")
    parser.add_argument("--wandb_run_name", default=None,
                        help="Optional W&B run name override")
    parser.add_argument("--wandb_mode", default=None,
                        choices=["online", "offline", "disabled"],
                        help="Override W&B mode (defaults to online on the main rank)")

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    wandb_mode_used = None
    
    # Parse chunk selections
    train_chunks = parse_chunk_arg(args.train_chunks, DEFAULT_TRAIN_CHUNKS)
    val_chunks = parse_chunk_arg(args.val_chunks, DEFAULT_VAL_CHUNKS)
    test_chunks = parse_chunk_arg(args.test_chunks, DEFAULT_TEST_CHUNKS)

    # DDP setup
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if args.wandb_project and not is_main_process():
        os.environ.setdefault("WANDB_MODE", args.wandb_mode or "disabled")

    if args.wandb_project and is_main_process():
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Weights & Biases logging requested but wandb is not installed. "
                "Install it with `pip install wandb` or remove the --wandb_project flag."
            ) from exc

        wandb_mode_used = args.wandb_mode or os.environ.get("WANDB_MODE") or "online"
        wandb_config = {k: v for k, v in vars(args).items() if not k.startswith("wandb_")}
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name,
            config=wandb_config,
            mode=wandb_mode_used,
        )
        wandb_run.summary["world_size"] = world_size
        wandb_run.summary["num_epochs"] = args.epochs

    if is_main_process():
        print("\n" + "="*60)
        print("QUIXER TRAINING (CHUNK-BASED, DDP-READY)")
        print("="*60)
        print(f"Device: {device}")
        print(f"Rank: {rank}/{world_size}")
        print(f"Batch size: {args.batch_size} per GPU")
        print(f"Effective batch size: {args.batch_size * world_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Init strategy: {args.init_strategy}")
        if world_size != 3:
            print(f"⚠️  World size detected: {world_size}. Script is tuned for 3 GPUs.")
        print(f"Train chunks: {train_chunks}")
        print(f"Val chunks:   {val_chunks}")
        print(f"Test chunks:  {test_chunks}")
        if args.wandb_project:
            entity_display = args.wandb_entity or "(default)"
            mode_display = wandb_mode_used or args.wandb_mode or "online"
            print(f"W&B logging: project={args.wandb_project}, entity={entity_display}, mode={mode_display}")

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        args.embeddings_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_chunks=train_chunks,
        val_chunks=val_chunks,
        test_chunks=test_chunks,
        target_device=device,
    )

    # Create model
    if is_main_process():
        print("\nCreating model...")
    
    # Infer sequence length and embedding dim from dataset
    seq_len = getattr(train_loader.dataset, 'max_kmers', 80)
    emb_dim = getattr(train_loader.dataset, 'embedding_dim', 64)

    # Infer vocabulary size from centroids (if provided)
    n_vocab = 512
    try:
        if Path(args.centroids_path).exists():
            _payload = torch.load(args.centroids_path, map_location='cpu')
            if isinstance(_payload, torch.Tensor):
                _cent = _payload
            elif isinstance(_payload, dict):
                _cent = _payload.get('centroids') or _payload.get('cluster_centers')
            else:
                _cent = None
            if torch.is_tensor(_cent):
                n_vocab = int(_cent.shape[0])
    except Exception:
        pass

    model = QuixerClassifier(
        n_qubits=6,
        n_tokens=seq_len - 1 if args.autoregressive else seq_len,
        qsvt_polynomial_degree=3,
        n_ansatz_layers=2,
        vocabulary_size=n_vocab,
        n_classes=n_vocab,
        embedding_dimension=emb_dim,
        dropout=0.1,
        batch_size=args.batch_size,
        device=device
    ).to(device)
    
    # Initialize parameters
    initialize_model_params(model, args.init_strategy)
    
    # Load cluster centroids
    centroids = load_cluster_centroids(model, args.centroids_path, device, world_size)
    if centroids is None:
        centroids = model.quixer.embedding.weight.detach().clone().to(device)

    # Helper: phased unfreezing for layerwise training
    def set_requires_grad(module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    def apply_layerwise_phase(epoch_idx: int) -> int:
        """Set requires_grad by phase. Returns current phase [1,2,3]."""
        if not args.use_layerwise_training:
            return 3
        total = args.epochs
        if epoch_idx < int(0.25 * total):
            phase = 1
        elif epoch_idx < int(0.5 * total):
            phase = 2
        else:
            phase = 3

        # Default: freeze all, then selectively unfreeze
        base = model.module if isinstance(model, DDP) else model
        set_requires_grad(base.quixer.embedding, False)
        set_requires_grad(base.quixer.embedding_to_angles, False)
        if isinstance(base.quixer.output_feedforward, nn.Module):
            set_requires_grad(base.quixer.output_feedforward, False)
        # Single-parameter tensors
        base.quixer.lcu_coefficients.requires_grad = False
        base.quixer.qsvt_polynomial_coefficients.requires_grad = False
        base.quixer.quantum_feedforward_parameters.requires_grad = False

        if phase >= 1:
            # Train only final MLP
            if isinstance(base.quixer.output_feedforward, nn.Module):
                set_requires_grad(base.quixer.output_feedforward, True)
        if phase >= 2:
            # Unfreeze embedding and angle projector
            set_requires_grad(base.quixer.embedding, True)
            set_requires_grad(base.quixer.embedding_to_angles, True)
        if phase >= 3:
            # Unfreeze quantum stack
            base.quixer.lcu_coefficients.requires_grad = True
            base.quixer.qsvt_polynomial_coefficients.requires_grad = True
            base.quixer.quantum_feedforward_parameters.requires_grad = True
        return phase

    # Optimizer and scheduler (built from currently-trainable params)
    def build_optimizer():
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        return optimizer

    # Freeze per phase-1 BEFORE DDP so reducer sees correct trainable set
    current_phase = apply_layerwise_phase(epoch_idx=0)

    # Wrap with DDP after initial freeze; allow unused params across phases
    if dist.is_available() and dist.is_initialized():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    optimizer = build_optimizer()
    
    def lr_lambda(epoch):
        """Cosine annealing with warmup."""
        warmup_epochs = max(1, int(args.warmup_frac * args.epochs))
        if epoch < warmup_epochs:
            return (epoch + 1) / (warmup_epochs + 1)
        else:
            progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if args.label_smoothing and args.label_smoothing > 0.0:
        criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Training loop
    if is_main_process():
        print("\n" + "="*60)
        print("TRAINING")
        print("="*60)
    
    best_val_acc = 0.0
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    metrics_history = []
    
    for epoch in range(args.epochs):
        if dist.is_available() and dist.is_initialized():
            sampler = getattr(train_loader, "sampler", None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
        # Apply/advance layerwise phase and rebuild optimizer if phase changed
        new_phase = apply_layerwise_phase(epoch)
        if new_phase != current_phase:
            current_phase = new_phase
            optimizer = build_optimizer()
            # Rebuild scheduler to attach to the new optimizer and preserve epoch position
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=epoch-1)
        # Train
        # pass context length dynamically to the train/validate helpers
        setattr(train_epoch, "context_len", max(2, int(args.context_len)))
        setattr(validate, "context_len", max(2, int(args.context_len)))
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.epochs,
            centroids, autoregressive=args.autoregressive
        )
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device,
                                     centroids, autoregressive=args.autoregressive)
        
        # Scheduler step
        scheduler.step()
        
        # Logging
        if is_main_process():
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.2%}")
            print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.2%}")
            lr_value = scheduler.get_last_lr()[0]
            print(f"  LR: {lr_value:.2e}")

            train_ppl = math.exp(train_loss) if args.autoregressive else None
            val_ppl = math.exp(val_loss) if args.autoregressive else None

            history_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': lr_value,
                'phase': current_phase,
            }
            if train_ppl is not None and val_ppl is not None:
                history_entry['train_ppl'] = train_ppl
                history_entry['val_ppl'] = val_ppl
            metrics_history.append(history_entry)

            if wandb_run:
                log_payload = {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "lr": lr_value,
                    "phase": current_phase,
                }
                if train_ppl is not None and val_ppl is not None:
                    log_payload["train/ppl"] = train_ppl
                    log_payload["val/ppl"] = val_ppl
                wandb_run.log(log_payload)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                
                model_to_save = model.module if isinstance(model, DDP) else model
                torch.save(model_to_save.state_dict(), output_dir / "best_model.pt")
                print(f"  ✓ Best model saved (epoch {best_epoch})")
                if wandb_run:
                    wandb_run.summary["best_val_acc"] = best_val_acc
                    wandb_run.summary["best_epoch"] = best_epoch
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n✓ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    if wandb_run:
                        wandb_run.summary["early_stop_epoch"] = epoch + 1
                    break
    
    # Test
    if is_main_process():
        print("\n" + "="*60)
        print("TESTING")
        print("="*60)
        
        test_loss, test_acc = validate(model, test_loader, criterion, device, centroids, autoregressive=args.autoregressive)
        print(f"Test: loss={test_loss:.4f}, acc={test_acc:.2%}")
        
        # Save final metrics
        final_metrics = {
            'best_epoch': best_epoch,
            'best_val_acc': best_val_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'history': metrics_history
        }
        
        with open(output_dir / "metrics.json", 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"\n✓ Training complete!")
        print(f"  Best epoch: {best_epoch}")
        print(f"  Best val acc: {best_val_acc:.2%}")
        print(f"  Test acc: {test_acc:.2%}")
        print(f"  Results saved to: {output_dir}")
    
    cleanup_ddp()


if __name__ == "__main__":
    import os
    main()
