import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import math


# =====================================================
# Dataset with Mosaic Augmentation
# =====================================================
class YOLODataset(Dataset):
    """YOLO Dataset with Mosaic Augmentation"""

    def __init__(self, img_dir, label_dir, img_size=640, augment=False, mosaic=0.5):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        self.mosaic_prob = mosaic

        self.images = sorted(
            list(self.img_dir.glob("*.jpg")) +
            list(self.img_dir.glob("*.png")) +
            list(self.img_dir.glob("*.jpeg"))
        )
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.img_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.augment and np.random.rand() < self.mosaic_prob:
            return self.load_mosaic(idx)
        else:
            return self.load_single(idx)

    def load_single(self, idx):
        """Load single image and labels"""
        img_path = self.images[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x, y, w, h = map(float, parts)
                        labels.append([cls, x, y, w, h])

        labels = np.array(labels, dtype=np.float32) if len(labels) > 0 else np.zeros((0, 5), dtype=np.float32)
        
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        labels = torch.from_numpy(labels)

        return img, labels, str(img_path)

    def _load_image_labels(self, idx):
        """Load raw image and labels"""
        img_path = self.images[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((640, 640, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        labels.append(list(map(float, parts)))

        return img, np.array(labels, dtype=np.float32) if len(labels) > 0 else np.zeros((0, 5), dtype=np.float32)

    def load_mosaic(self, idx):
        """Load 4 images in mosaic pattern"""
        img_size = self.img_size
        yc, xc = [int(np.random.uniform(img_size * 0.5, img_size * 1.5)) for _ in range(2)]

        indices = [idx] + np.random.randint(0, len(self.images), 3).tolist()
        img4 = np.full((img_size * 2, img_size * 2, 3), 114, dtype=np.uint8)
        labels4 = []

        for i, index in enumerate(indices):
            img, labels = self._load_image_labels(index)
            h, w = img.shape[:2]

            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, img_size * 2), yc
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(yc + h, img_size * 2)
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, img_size * 2), min(yc + h, img_size * 2)

            img_resized = cv2.resize(img, (x2a - x1a, y2a - y1a))
            img4[y1a:y2a, x1a:x2a] = img_resized

            if len(labels) > 0:
                labels_copy = labels.copy()
                labels_copy[:, 1:] *= img_size
                labels_copy[:, 1] += x1a
                labels_copy[:, 2] += y1a
                labels4.append(labels_copy)

        if len(labels4) > 0:
            labels4 = np.concatenate(labels4, 0)
            labels4[:, 1:] /= (img_size * 2)
            labels4[:, [1, 3]] = labels4[:, [1, 3]].clip(0, 1)
            labels4[:, [2, 4]] = labels4[:, [2, 4]].clip(0, 1)

        img4 = cv2.resize(img4, (img_size, img_size))
        img4 = torch.from_numpy(img4).permute(2, 0, 1).float() / 255.0
        labels4 = torch.from_numpy(labels4) if len(labels4) > 0 else torch.zeros((0, 5))

        return img4, labels4, ""


# =====================================================
# Collate Function
# =====================================================
def collate_fn(batch):
    """Custom collate function for YOLO dataloader"""
    images, labels, paths = zip(*batch)
    images = torch.stack(images, 0)

    targets = []
    for i, label in enumerate(labels):
        if label.numel() > 0:
            batch_idx = torch.full((label.shape[0], 1), i)
            targets.append(torch.cat((batch_idx, label), 1))

    targets = torch.cat(targets, 0) if len(targets) > 0 else torch.zeros((0, 6))
    return images, targets, paths


# =====================================================
# EMA (Exponential Moving Average)
# =====================================================
class ModelEMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.9999):
        self.ema = deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for e, m in zip(self.ema.parameters(), model.parameters()):
                e.mul_(self.decay).add_(m, alpha=1 - self.decay)

    def eval_model(self):
        return self.ema


# =====================================================
# Warmup + Cosine Learning Rate Scheduler
# =====================================================
class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_start, lr_max, lr_min):
        self.opt = optimizer
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch <= self.warmup:
            lr = self.lr_start + (self.lr_max - self.lr_start) * self.epoch / self.warmup
        else:
            t = (self.epoch - self.warmup) / (self.total - self.warmup)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * t))

        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr


# =====================================================
# Training Functions
# =====================================================
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, 
                    ema=None, grad_clip=10.0, scaler=None):
    """Train for one epoch"""
    model.train()
    
    losses = {'box': [], 'cls': [], 'dfl': [], 'total': []}
    
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [TRAIN]", ncols=120)
    
    for batch_idx, (imgs, targets, _) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                loss_dict = criterion(preds, targets)
                loss = loss_dict['total']
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(imgs)
            loss_dict = criterion(preds, targets)
            loss = loss_dict['total']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        if ema is not None:
            ema.update(model)
        
        for k in losses:
            losses[k].append(loss_dict[k].item())
        
        avg_window = min(50, len(losses['total']))
        pbar.set_postfix({
            'box': f"{np.mean(losses['box'][-avg_window:]):.3f}",
            'cls': f"{np.mean(losses['cls'][-avg_window:]):.3f}",
            'dfl': f"{np.mean(losses['dfl'][-avg_window:]):.3f}",
            'total': f"{np.mean(losses['total'][-avg_window:]):.3f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
    
    return {k: np.mean(v) for k, v in losses.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    
    losses = {'box': [], 'cls': [], 'dfl': [], 'total': []}
    
    pbar = tqdm(loader, desc="[VAL]", ncols=120)
    
    for imgs, targets, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        preds = model(imgs)
        loss_dict = criterion(preds, targets)
        
        for k in losses:
            losses[k].append(loss_dict[k].item())
        
        pbar.set_postfix({
            'box': f"{np.mean(losses['box']):.3f}",
            'cls': f"{np.mean(losses['cls']):.3f}",
            'dfl': f"{np.mean(losses['dfl']):.3f}",
            'total': f"{np.mean(losses['total']):.3f}"
        })
    
    return {k: np.mean(v) for k, v in losses.items()}


def train_yolov8(model, train_loader, val_loader, criterion, optimizer, 
                 scheduler, config, device):
    """Complete training loop"""
    
    best_loss = float('inf')
    ema = ModelEMA(model, decay=config.get('ema_decay', 0.9999)) if config.get('ema', True) else None
    scaler = torch.cuda.amp.GradScaler() if config.get('amp', True) else None
    
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, config['epochs'] + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, ema=ema, grad_clip=config.get('grad_clip', 10.0), scaler=scaler
        )
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Validate
        if val_loader is not None and epoch % config.get('val_interval', 1) == 0:
            val_model = ema.eval_model() if ema is not None else model
            val_loss = validate(val_model, val_loader, criterion, device)
            
            print(f"\nEpoch {epoch}/{config['epochs']}:")
            print(f"  Train - Box: {train_loss['box']:.4f} | Cls: {train_loss['cls']:.4f} | DFL: {train_loss['dfl']:.4f} | Total: {train_loss['total']:.4f}")
            print(f"  Val   - Box: {val_loss['box']:.4f} | Cls: {val_loss['cls']:.4f} | DFL: {val_loss['dfl']:.4f} | Total: {val_loss['total']:.4f}")
            
            # Save best model
            if val_loss['total'] < best_loss:
                best_loss = val_loss['total']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': ema.ema.state_dict() if ema else None,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss['total'],
                }, save_dir / 'best.pt')
                print(f"  ✅ Saved best model (loss: {best_loss:.4f})")
        
        # Save checkpoint
        if epoch % config.get('save_interval', 10) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.ema.state_dict() if ema else None,
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_dir / f'epoch_{epoch}.pt')
    
    # Save final model
    torch.save({
        'epoch': config['epochs'],
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema.ema.state_dict() if ema else None,
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_dir / 'last.pt')
    
    print(f"\n✅ Training complete! Best loss: {best_loss:.4f}")