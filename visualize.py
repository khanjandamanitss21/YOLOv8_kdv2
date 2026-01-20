"""
YOLOv8 Visualization Script for Aadhaar Card Detection
Visualizes ground truth vs predictions in YOLO style

Usage:
    python visualize.py --weights runs/train/exp/best.pt --data data.yaml --num-images 3
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm

from yolov8_model import YOLOv8
from train_utils import YOLODataset, collate_fn
from torch.utils.data import DataLoader


def load_model(weights_path, nc, device):
    """Load trained model"""
    checkpoint = torch.load(weights_path, map_location=device)
    
    # Create model with correct number of classes
    model = YOLOv8(nc=nc, depth=0.67, width=0.75).to(device)
    
    # Load weights (handle both EMA and regular checkpoints)
    if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
        model.load_state_dict(checkpoint['ema_state_dict'])
        print("✅ Loaded EMA weights")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded model weights")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded weights directly")
    
    model.eval()
    return model


def decode_boxes(pred_dist, anchors, strides, reg_max=16):
    """
    Decode distance predictions to boxes
    
    Args:
        pred_dist: [B, 8400, 64] distance predictions
        anchors: [8400, 2] anchor points
        strides: [8400] stride values
        reg_max: 16
    
    Returns:
        boxes: [B, 8400, 4] in xyxy format (pixel coordinates)
    """
    B, A, _ = pred_dist.shape
    device = pred_dist.device
    
    # Reshape to [B, A, 4, reg_max]
    pred = pred_dist.view(B, A, 4, reg_max)
    prob = pred.softmax(-1)
    
    # Project to get distances in stride units
    proj = torch.arange(reg_max, device=device, dtype=pred.dtype)
    dist_stride_units = (prob * proj.view(1, 1, 1, -1)).sum(-1)  # [B, A, 4]
    
    # Get anchors and strides
    anchors = anchors.unsqueeze(0)  # [1, A, 2]
    strides = strides.unsqueeze(0).unsqueeze(-1)  # [1, A, 1]
    
    # Scale by stride to get pixel distances
    dist_pixels = dist_stride_units * strides  # [B, A, 4]
    
    # Convert to xyxy
    x1 = anchors[..., 0:1] - dist_pixels[..., 0:1]
    y1 = anchors[..., 1:2] - dist_pixels[..., 1:2]
    x2 = anchors[..., 0:1] + dist_pixels[..., 2:3]
    y2 = anchors[..., 1:2] + dist_pixels[..., 3:4]
    
    boxes = torch.cat([x1, y1, x2, y2], -1)
    boxes = boxes.clamp(0, 640)
    
    return boxes


def decode_predictions(preds, anchors, strides, conf_thres=0.25, iou_thres=0.45, nc=5):
    """
    Decode model predictions to boxes, scores, classes
    
    Args:
        preds: List of [P3, P4, P5] predictions from model
        anchors: [8400, 2] anchor points
        strides: [8400] stride values
        conf_thres: Confidence threshold
        iou_thres: NMS IoU threshold
        nc: Number of classes
    
    Returns:
        pred_boxes: List of [N, 4] boxes per image
        pred_scores: List of [N] scores per image
        pred_classes: List of [N] class indices per image
    """
    # Concatenate predictions from all scales
    cls_out, reg_out = [], []
    
    for p in preds:
        # Split into regression (64 channels) and classification (nc channels)
        d, c = p.split((64, nc), 1)
        cls_out.append(c.flatten(2).permute(0, 2, 1))  # [B, H*W, nc]
        reg_out.append(d.flatten(2).permute(0, 2, 1))  # [B, H*W, 64]
    
    pred_scores = torch.cat(cls_out, 1).sigmoid()  # [B, 8400, nc]
    pred_dist = torch.cat(reg_out, 1)              # [B, 8400, 64]
    
    # Decode boxes
    pred_boxes = decode_boxes(pred_dist, anchors, strides)  # [B, 8400, 4]
    
    # Get max class score and class index
    class_scores, class_idx = pred_scores.max(2)  # [B, 8400]
    
    # Filter by confidence and apply NMS per image
    batch_size = pred_boxes.shape[0]
    final_boxes, final_scores, final_classes = [], [], []
    
    for i in range(batch_size):
        # Filter by confidence
        mask = class_scores[i] > conf_thres
        
        boxes = pred_boxes[i][mask]
        scores = class_scores[i][mask]
        classes = class_idx[i][mask]
        
        if len(boxes) == 0:
            final_boxes.append(torch.zeros((0, 4)))
            final_scores.append(torch.zeros(0))
            final_classes.append(torch.zeros(0))
            continue
        
        # NMS
        try:
            import torchvision
            keep = torchvision.ops.nms(boxes, scores, iou_thres)
        except:
            # Fallback if torchvision not available
            keep = torch.arange(len(boxes))
        
        final_boxes.append(boxes[keep].cpu().numpy())
        final_scores.append(scores[keep].cpu().numpy())
        final_classes.append(classes[keep].cpu().numpy())
    
    return final_boxes, final_scores, final_classes


def plot_one_box(box, img, color=None, label=None, line_thickness=2):
    """
    Plot one bounding box on image
    
    Args:
        box: [x1, y1, x2, y2]
        img: numpy array
        color: RGB color tuple
        label: string label
        line_thickness: int
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], 
                    thickness=tf, lineType=cv2.LINE_AA)


def visualize_batch(model, dataloader, device, num_images=3, conf_thres=0.25, 
                    iou_thres=0.45, class_names=None, save_dir='runs/visualize'):
    """
    Visualize ground truth vs predictions for validation images
    
    Args:
        model: YOLOv8 model
        dataloader: validation dataloader
        device: torch device
        num_images: number of images to visualize
        conf_thres: confidence threshold
        iou_thres: NMS IoU threshold
        class_names: dict of class names {0: 'name1', 1: 'name2', ...}
        save_dir: directory to save visualizations
    """
    if class_names is None:
        class_names = {0: 'class_0', 1: 'class_1', 2: 'class_2', 3: 'class_3', 4: 'class_4'}
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    nc = len(class_names)
    
    # Generate anchors and strides (same as in loss function)
    anchors, strides = make_anchors(device)
    
    # Colors for each class (BGR for OpenCV)
    # Distinct colors for Aadhaar fields
    colors = {
        0: (255, 0, 0),     # Blue - aadhaar_number
        1: (0, 255, 0),     # Green - address
        2: (0, 0, 255),     # Red - date_of_birth
        3: (255, 255, 0),   # Cyan - gender
        4: (255, 0, 255),   # Magenta - name
    }
    
    img_count = 0
    
    print(f"\n{'='*60}")
    print(f"Starting visualization with conf_thres={conf_thres}, iou_thres={iou_thres}")
    print(f"{'='*60}\n")
    
    with torch.no_grad():
        for batch_idx, (images, targets, paths) in enumerate(tqdm(dataloader, desc="Visualizing")):
            if img_count >= num_images:
                break
            
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            preds = model(images)
            
            # Decode predictions
            pred_boxes, pred_scores, pred_classes = decode_predictions(
                preds, anchors, strides, conf_thres=conf_thres, iou_thres=iou_thres, nc=nc
            )
            
            # Process each image in batch
            for i in range(images.shape[0]):
                if img_count >= num_images:
                    break
                
                # Get image
                img = images[i].cpu().permute(1, 2, 0).numpy()  # [H, W, C]
                img = (img * 255).astype(np.uint8).copy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                h, w = img.shape[:2]
                
                # Create side-by-side visualization
                vis_gt = img.copy()
                vis_pred = img.copy()
                
                # Draw ground truth
                batch_targets = targets[targets[:, 0] == i]  # Get targets for this image
                gt_count = len(batch_targets)
                
                for target in batch_targets:
                    _, cls, x_center, y_center, box_w, box_h = target.cpu().numpy()
                    
                    # Convert normalized to pixel coordinates
                    x_center *= w
                    y_center *= h
                    box_w *= w
                    box_h *= h
                    
                    x1 = int(x_center - box_w / 2)
                    y1 = int(y_center - box_h / 2)
                    x2 = int(x_center + box_w / 2)
                    y2 = int(y_center + box_h / 2)
                    
                    cls_id = int(cls)
                    label = class_names.get(cls_id, f"class_{cls_id}")
                    color = colors.get(cls_id, (255, 255, 255))
                    
                    plot_one_box([x1, y1, x2, y2], vis_gt, color=color, label=label, line_thickness=3)
                
                # Draw predictions
                pred_count = 0
                if i < len(pred_boxes) and len(pred_boxes[i]) > 0:
                    pred_count = len(pred_boxes[i])
                    for box, score, cls_id in zip(pred_boxes[i], pred_scores[i], pred_classes[i]):
                        x1, y1, x2, y2 = box
                        cls_id = int(cls_id)
                        label = f"{class_names.get(cls_id, f'class_{cls_id}')} {score:.2f}"
                        color = colors.get(cls_id, (255, 255, 255))
                        
                        plot_one_box([x1, y1, x2, y2], vis_pred, color=color, label=label, line_thickness=3)
                
                # Combine images side by side
                combined = np.hstack([vis_gt, vis_pred])
                
                # Add titles and info
                title_height = 60
                info_img = np.zeros((title_height, combined.shape[1], 3), dtype=np.uint8)
                
                # Ground Truth title
                cv2.putText(info_img, "Ground Truth", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(info_img, f"Objects: {gt_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Predictions title
                cv2.putText(info_img, "Predictions", (w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(info_img, f"Detections: {pred_count}", (w + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Stack title on top
                combined = np.vstack([info_img, combined])
                
                # Save
                save_path = save_dir / f"val_batch{batch_idx}_img{i}.jpg"
                cv2.imwrite(str(save_path), combined)
                
                print(f"✅ Saved: {save_path.name} | GT: {gt_count} | Pred: {pred_count}")
                
                img_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ All visualizations saved to: {save_dir}")
    print(f"{'='*60}\n")
    
    # Print legend
    print("Class Color Legend:")
    print("-" * 40)
    for cls_id, name in class_names.items():
        color_name = {
            (255, 0, 0): "Blue",
            (0, 255, 0): "Green",
            (0, 0, 255): "Red",
            (255, 255, 0): "Cyan",
            (255, 0, 255): "Magenta",
        }.get(colors.get(cls_id), "White")
        print(f"  {cls_id}: {name:20s} - {color_name}")
    print("-" * 40)


def make_anchors(device):
    """Generate anchor points and strides (same as model)"""
    anchors = []
    strides_list = []
    
    for stride, size in zip([8, 16, 32], [80, 40, 20]):
        y, x = torch.meshgrid(
            torch.arange(size, device=device),
            torch.arange(size, device=device),
            indexing="ij"
        )
        xy = torch.stack([x, y], -1).view(-1, 2).float()
        xy = (xy + 0.5) * stride  # Anchor centers in pixels
        anchors.append(xy)
        strides_list.extend([stride] * (size * size))
    
    anchors = torch.cat(anchors)  # [8400, 2]
    strides = torch.tensor(strides_list, device=device, dtype=torch.float32)  # [8400]
    
    return anchors, strides


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Visualization for Aadhaar Detection')
    parser.add_argument('--weights', type=str, required=True, help='path to weights file (e.g., runs/train/exp/best.pt)')
    parser.add_argument('--data', type=str, required=True, help='path to data.yaml')
    parser.add_argument('--num-images', type=int, default=3, help='number of images to visualize')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for dataloader')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', type=str, default='0', help='cuda device (e.g., 0 or cpu)')
    parser.add_argument('--save-dir', type=str, default='runs/visualize', help='directory to save visualizations')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    
    # Load data config
    with open(args.data) as f:
        data_config = yaml.safe_load(f)
    
    # Get class names
    class_names = data_config.get('names', {})
    nc = data_config.get('nc', len(class_names))
    
    print("\n" + "="*60)
    print("YOLOv8 Aadhaar Card Detection - Visualization")
    print("="*60)
    print(f"Weights:        {args.weights}")
    print(f"Device:         {device}")
    print(f"Classes:        {nc}")
    print(f"Class names:    {class_names}")
    print(f"Conf threshold: {args.conf_thres}")
    print(f"IoU threshold:  {args.iou_thres}")
    print(f"Save directory: {args.save_dir}")
    print("="*60 + "\n")
    
    # Load model
    print("Loading model...")
    model = load_model(args.weights, nc, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M\n")
    
    # Create validation dataloader
    print("Loading validation data...")
    root = Path(data_config['path'])
    val_img_dir = root / data_config['val']
    val_label_dir = root / data_config['val'].replace('images', 'labels')
    
    print(f"  Images:  {val_img_dir}")
    print(f"  Labels:  {val_label_dir}")
    
    val_dataset = YOLODataset(
        img_dir=val_img_dir,
        label_dir=val_label_dir,
        img_size=640,
        augment=False,
        mosaic=0.0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"  Dataset size: {len(val_dataset)} images")
    print(f"  Batches:      {len(val_loader)}\n")
    
    # Visualize
    visualize_batch(
        model=model,
        dataloader=val_loader,
        device=device,
        num_images=args.num_images,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        class_names=class_names,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()