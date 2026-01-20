"""
Complete YOLOv8 Training Script
Production-ready implementation with all optimizations

Usage:
    python train.py --data data.yaml --epochs 300 --batch 16
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import yaml
import os

# Import components
from yolov8_model import YOLOv8
from yolov8_loss import YOLOv8Loss
from train_utils import (
    YOLODataset, 
    collate_fn, 
    train_yolov8,
    WarmupCosineScheduler
)


def create_dataloaders(data_config, batch_size=16, workers=4):
    """Create train and validation dataloaders"""
    root = Path(data_config['path'])
    
    # Resolve paths
    train_img_dir = root / data_config['train']
    train_label_dir = root / data_config['train'].replace('images', 'labels')
    val_img_dir = root / data_config['val']
    val_label_dir = root / data_config['val'].replace('images', 'labels')
    
    print(f"  Train images: {train_img_dir}")
    print(f"  Train labels: {train_label_dir}")
    print(f"  Val images: {val_img_dir}")
    print(f"  Val labels: {val_label_dir}")
    
    # Create datasets
    train_dataset = YOLODataset(
        img_dir=train_img_dir,
        label_dir=train_label_dir,
        img_size=data_config.get('img_size', 640),
        augment=True,
        mosaic=0.5
    )

    val_dataset = YOLODataset(
        img_dir=val_img_dir,
        label_dir=val_label_dir,
        img_size=data_config.get('img_size', 640),
        augment=False,
        mosaic=0.0
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if workers > 0 else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if workers > 0 else False
    )
    
    return train_loader, val_loader


def build_optimizer(model, lr=0.001, momentum=0.937, weight_decay=0.0005):
    """Build optimizer with parameter groups"""
    
    g = [], [], []  # parameter groups: conv weights, bn weights, biases
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
    
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == 'bias':
                g[2].append(p)
            elif p_name == 'weight' and isinstance(v, bn):
                g[1].append(p)
            else:
                g[0].append(p)
    
    optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    optimizer.add_param_group({'params': g[0], 'weight_decay': weight_decay})
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})
    
    del g
    return optimizer


def main():
    # ==================== Parse Arguments ====================
    parser = argparse.ArgumentParser(description='YOLOv8 Training')
    parser.add_argument('--data', type=str, required=True, help='data.yaml path')
    parser.add_argument('--epochs', type=int, default=300, help='training epochs')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--device', type=str, default='0', help='cuda device')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers')
    parser.add_argument('--project', type=str, default='runs/train', help='save directory')
    parser.add_argument('--name', type=str, default='exp', help='experiment name')
    parser.add_argument('--resume', type=str, default='', help='resume checkpoint')
    parser.add_argument('--model-size', type=str, default='m', choices=['n', 's', 'm', 'l', 'x'],
                        help='model size (n/s/m/l/x)')
    
    args = parser.parse_args()
    
    # ==================== Load Config ====================
    with open(args.data) as f:
        data_config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    # Create save directory
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'args.yaml', 'w') as f:
        yaml.dump(vars(args), f)
    
    print("\n" + "="*80)
    print("🚀 YOLOv8 Training Configuration")
    print("="*80)
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.img_size}")
    print(f"  Model size: YOLOv8{args.model_size}")
    print(f"  Workers: {args.workers}")
    print(f"  Save dir: {save_dir}")
    print("="*80 + "\n")
    
    # ==================== Model ====================
    print("🔨 Building model...")
    
    # Model size configurations
    model_configs = {
        'n': {'depth': 0.33, 'width': 0.25},  # Nano
        's': {'depth': 0.33, 'width': 0.50},  # Small
        'm': {'depth': 0.67, 'width': 0.75},  # Medium
        'l': {'depth': 1.00, 'width': 1.00},  # Large
        'x': {'depth': 1.00, 'width': 1.25},  # XLarge
    }
    
    config = model_configs[args.model_size]
    model = YOLOv8(
        nc=data_config['nc'],
        depth=config['depth'],
        width=config['width']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: YOLOv8{args.model_size}")
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M\n")
    
    # ==================== Loss ====================
    print("📊 Building loss function...")
    criterion = YOLOv8Loss(model, nc=data_config['nc'], reg_max=16)
    print("  Loss: Task-Aligned Assigner + CIoU + DFL\n")
    
    # ==================== Data ====================
    print("📁 Loading datasets...")
    train_loader, val_loader = create_dataloaders(
        data_config,
        batch_size=args.batch,
        workers=args.workers
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}\n")
    
    # ==================== Optimizer & Scheduler ====================
    print("⚙️  Building optimizer and scheduler...")
    
    # Learning rate settings (CRITICAL for good performance)
    lr_start = 1e-5
    lr_max = 0.001  # Lower LR for stability
    lr_min = 1e-5
    warmup_epochs = 3
    
    optimizer = build_optimizer(model, lr=lr_max, momentum=0.937, weight_decay=0.0005)
    
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=args.epochs,
        lr_start=lr_start,
        lr_max=lr_max,
        lr_min=lr_min
    )
    
    print(f"  Optimizer: SGD with Nesterov momentum")
    print(f"  LR schedule: Warmup ({warmup_epochs} epochs) + Cosine")
    print(f"  LR range: {lr_start:.6f} → {lr_max:.6f} → {lr_min:.6f}\n")
    
    # ==================== Resume Training ====================
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        scheduler.epoch = start_epoch
        print(f"  Resumed at epoch {start_epoch}\n")
    
    # ==================== Training Config ====================
    train_config = {
        'epochs': args.epochs,
        'warmup_epochs': warmup_epochs,
        'amp': True,
        'ema': True,
        'ema_decay': 0.9999,
        'grad_clip': 10.0,
        'val_interval': 1,
        'save_interval': 10,
        'save_dir': save_dir,
    }
    
    # ==================== Training ====================
    print("="*80)
    print("🚀 Starting Training")
    print("="*80)
    print(f"  Mixed Precision: {train_config['amp']}")
    print(f"  EMA: {train_config['ema']}")
    print(f"  Gradient Clipping: {train_config['grad_clip']}")
    print(f"  Validation Interval: {train_config['val_interval']} epoch(s)")
    print("="*80 + "\n")
    
    try:
        train_yolov8(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=train_config,
            device=device
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        torch.save({
            'epoch': scheduler.epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_dir / 'interrupted.pt')
        print(f"  Saved checkpoint to {save_dir / 'interrupted.pt'}")
    
    print("\n" + "="*80)
    print("✅ Training Complete!")
    print("="*80)
    print(f"  Results saved to: {save_dir}")
    print(f"  Best model: {save_dir / 'best.pt'}")
    print(f"  Last model: {save_dir / 'last.pt'}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()