import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskAlignedAssigner:
    """Task Aligned Assigner for YOLOv8"""
    def __init__(self, topk=10, alpha=0.5, beta=6.0):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anchors, gt_labels, gt_bboxes, mask_gt):
        """
        Assign ground truth to anchors.
        
        Args:
            pd_scores: [B, A, C] predicted class scores
            pd_bboxes: [B, A, 4] predicted boxes (in pixel coordinates)
            anchors: [A, 2] anchor points (in pixel coordinates)
            gt_labels: [B, G] ground truth labels
            gt_bboxes: [B, G, 4] ground truth boxes (in pixel coordinates)
            mask_gt: [B, G] valid gt mask
        
        Returns:
            target_labels: [B, A] assigned class labels
            target_bboxes: [B, A, 4] assigned boxes
            target_scores: [B, A, C] assigned scores
            fg_mask: [B, A] foreground mask
        """
        B, A, C = pd_scores.shape
        G = gt_bboxes.shape[1]
        device = pd_scores.device

        # Early return if no ground truth
        if G == 0 or not mask_gt.any():
            return (
                torch.zeros((B, A), dtype=torch.long, device=device),
                torch.zeros((B, A, 4), device=device),
                torch.zeros((B, A, C), device=device),
                torch.zeros((B, A), dtype=torch.bool, device=device),
            )

        # Compute IoU [B, G, A]
        iou = self.iou(gt_bboxes, pd_bboxes)
        
        # Get classification scores - use max across classes
        cls_score = pd_scores.max(dim=-1)[0]  # [B, A]
        cls_score = cls_score.unsqueeze(1).expand(-1, G, -1)  # [B, G, A]
        cls_score = cls_score * mask_gt.unsqueeze(-1).float()
        
        # Compute alignment metric
        align_metric = (cls_score ** self.alpha) * (iou ** self.beta)
        align_metric = align_metric * mask_gt.unsqueeze(-1).float()

        # Select top-k anchors
        topk = min(self.topk, A)
        topk_values, topk_idx = align_metric.topk(topk, dim=-1)

        # Create positive mask
        mask_pos = torch.zeros((B, G, A), dtype=torch.bool, device=device)
        valid_mask = (mask_gt.unsqueeze(-1) & (topk_values > 0))
        
        for k in range(topk):
            idx = topk_idx[:, :, k].unsqueeze(-1)
            mask_pos.scatter_(2, idx, valid_mask[:, :, k:k+1])

        # Foreground mask
        fg_mask = mask_pos.sum(1) > 0  # [B, A]
        
        if not fg_mask.any():
            return (
                torch.zeros((B, A), dtype=torch.long, device=device),
                torch.zeros((B, A, 4), device=device),
                torch.zeros((B, A, C), device=device),
                torch.zeros((B, A), dtype=torch.bool, device=device),
            )
        
        # Match GTs to anchors
        masked_iou = iou * mask_pos.float()
        matched_gt = masked_iou.argmax(1)  # [B, A]

        # Gather targets
        target_labels = gt_labels.gather(1, matched_gt)
        target_labels = target_labels.clamp(0, C - 1)
        target_labels[~fg_mask] = 0
        
        gt_idx_expanded = matched_gt.unsqueeze(-1).expand(-1, -1, 4)
        target_bboxes = gt_bboxes.gather(1, gt_idx_expanded)

        # Create target scores with IoU quality
        target_scores = torch.zeros((B, A, C), device=device)
        iou_scores = masked_iou.gather(1, matched_gt.unsqueeze(1)).squeeze(1)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), iou_scores.unsqueeze(-1))
        target_scores *= fg_mask.unsqueeze(-1).float()

        return target_labels, target_bboxes, target_scores, fg_mask

    @staticmethod
    def iou(box1, box2, eps=1e-7):
        """Calculate IoU between boxes."""
        b1 = box1.unsqueeze(2)  # [B, G, 1, 4]
        b2 = box2.unsqueeze(1)  # [B, 1, A, 4]

        inter = (torch.min(b1[..., 2:], b2[..., 2:]) -
                 torch.max(b1[..., :2], b2[..., :2])).clamp(0).prod(-1)

        area1 = (b1[..., 2:] - b1[..., :2]).prod(-1)
        area2 = (b2[..., 2:] - b2[..., :2]).prod(-1)

        return inter / (area1 + area2 - inter + eps)


class BboxLoss(nn.Module):
    """Bounding Box + DFL Loss - COMPLETELY FIXED"""
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, pred_boxes, anchors, strides, target_boxes, target_scores, score_sum, fg_mask):
        """
        Compute bounding box and DFL loss
        
        CRITICAL FIX: Proper normalization of losses
        """
        if not fg_mask.any():
            device = pred_dist.device
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # Get number of foreground anchors for normalization
        num_fg = fg_mask.sum().item()
        
        # Extract weights from target scores (these are IoU values)
        weight = target_scores.sum(-1)[fg_mask]  # [num_fg]
        
        # CIoU loss - average over foreground anchors
        iou = self.ciou(pred_boxes[fg_mask], target_boxes[fg_mask])
        loss_box = (1.0 - iou).mean()  # Simple mean, not weighted sum!

        # DFL loss
        B, A = pred_boxes.shape[:2]
        
        # Expand strides to [B, A]
        strides_expanded = strides.unsqueeze(0).expand(B, -1)
        
        # Convert target boxes to distances
        target_ltrb = self.box2dist(anchors, target_boxes, strides_expanded)
        target_ltrb = target_ltrb.clamp(0.0, self.reg_max - 1.01)

        # Extract foreground
        pred_d = pred_dist[fg_mask].view(-1, 4, self.reg_max)
        target_d = target_ltrb[fg_mask]
        
        # DFL: interpolate between floor and ceil
        target_left = target_d.floor().long().clamp(0, self.reg_max - 1)
        target_right = (target_left + 1).clamp(0, self.reg_max - 1)
        weight_right = (target_d - target_left.float()).clamp(0, 1)
        weight_left = 1.0 - weight_right
        
        loss_dfl = 0.0
        for i in range(4):
            pred_i = pred_d[:, i, :]
            loss_left = F.cross_entropy(pred_i, target_left[:, i], reduction='none')
            loss_right = F.cross_entropy(pred_i, target_right[:, i], reduction='none')
            loss_dfl += (loss_left * weight_left[:, i] + loss_right * weight_right[:, i])
        
        loss_dfl = loss_dfl.mean() / 4.0  # Average over foreground and 4 sides

        return loss_box, loss_dfl

    @staticmethod
    def ciou(b1, b2, eps=1e-7):
        """Complete IoU loss"""
        # Intersection
        inter = (torch.min(b1[:, 2:], b2[:, 2:]) -
                 torch.max(b1[:, :2], b2[:, :2])).clamp(0).prod(-1)
        
        # Union
        area1 = ((b1[:, 2:] - b1[:, :2]).clamp(min=eps)).prod(-1)
        area2 = ((b2[:, 2:] - b2[:, :2]).clamp(min=eps)).prod(-1)
        union = area1 + area2 - inter + eps
        
        # IoU
        iou = inter / union
        
        # Smallest enclosing box
        cw = torch.max(b1[:, 2], b2[:, 2]) - torch.min(b1[:, 0], b2[:, 0])
        ch = torch.max(b1[:, 3], b2[:, 3]) - torch.min(b1[:, 1], b2[:, 1])
        c2 = cw ** 2 + ch ** 2 + eps
        
        # Center distance
        b1_center = (b1[:, :2] + b1[:, 2:]) / 2
        b2_center = (b2[:, :2] + b2[:, 2:]) / 2
        rho2 = ((b1_center - b2_center) ** 2).sum(-1)
        
        # Aspect ratio consistency
        w1 = (b1[:, 2] - b1[:, 0]).clamp(min=eps)
        h1 = (b1[:, 3] - b1[:, 1]).clamp(min=eps)
        w2 = (b2[:, 2] - b2[:, 0]).clamp(min=eps)
        h2 = (b2[:, 3] - b2[:, 1]).clamp(min=eps)
        
        v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
        
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)
        
        # CIoU
        ciou = iou - (rho2 / c2 + v * alpha)
        
        return ciou.clamp(min=-1.0, max=1.0)

    @staticmethod
    def box2dist(anchors, boxes, strides):
        """
        Convert boxes to distance format for DFL
        
        Args:
            anchors: [A, 2] anchor points (x, y) in PIXELS
            boxes: [B, A, 4] boxes in xyxy format (x1,y1,x2,y2) in PIXELS
            strides: [B, A] stride values (already expanded for batch)
        
        Returns:
            [B, A, 4] distances (left, top, right, bottom) in STRIDE UNITS
        """
        B, A = boxes.shape[:2]
        
        # Expand anchors for batch [B, A, 2]
        anchors_expanded = anchors.unsqueeze(0).expand(B, -1, -1)
        
        # Calculate distances in PIXELS first
        left = anchors_expanded[..., 0] - boxes[..., 0]      # anchor_x - x1
        top = anchors_expanded[..., 1] - boxes[..., 1]       # anchor_y - y1  
        right = boxes[..., 2] - anchors_expanded[..., 0]     # x2 - anchor_x
        bottom = boxes[..., 3] - anchors_expanded[..., 1]    # y2 - anchor_y
        
        # Stack [B, A, 4]
        dist_pixels = torch.stack([left, top, right, bottom], dim=-1)
        
        # CRITICAL: Divide by strides to normalize to DFL range
        # strides is [B, A], we need [B, A, 1] for broadcasting
        strides_expanded = strides.unsqueeze(-1)  # [B, A, 1]
        dist_stride_units = dist_pixels / strides_expanded
        
        return dist_stride_units


class YOLOv8Loss(nn.Module):
    """Complete YOLOv8 Loss Function"""
    def __init__(self, model, nc=80, reg_max=16):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        
        # Loss weights
        self.box_weight = 7.5
        self.cls_weight = 0.5
        self.dfl_weight = 1.5

        self.assigner = TaskAlignedAssigner(topk=10, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(reg_max)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        device = next(model.parameters()).device
        self.anchors, self.strides = self.make_anchors(device)

    def make_anchors(self, device):
        """Generate anchor points and strides"""
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

    def decode(self, pred_dist):
        """Decode distance predictions to boxes in pixel coordinates"""
        B, A, _ = pred_dist.shape
        device = pred_dist.device
        
        # Reshape to [B, A, 4, reg_max]
        pred = pred_dist.view(B, A, 4, self.reg_max)
        prob = pred.softmax(-1)
        
        # Project to get distances in stride units [B, A, 4]
        proj = torch.arange(self.reg_max, device=device, dtype=pred.dtype)
        dist_stride_units = (prob * proj.view(1, 1, 1, -1)).sum(-1)
        
        # Expand strides [1, A, 1]
        strides = self.strides.unsqueeze(0).unsqueeze(-1)  # [1, A, 1]
        
        # Scale distances by stride to get PIXEL distances
        dist_pixels = dist_stride_units * strides  # [B, A, 4]
        
        # Get anchors [1, A, 2]
        anchors = self.anchors.unsqueeze(0)
        
        # Convert from distances to box coordinates
        x1 = anchors[..., 0:1] - dist_pixels[..., 0:1]  # anchor_x - left
        y1 = anchors[..., 1:2] - dist_pixels[..., 1:2]  # anchor_y - top
        x2 = anchors[..., 0:1] + dist_pixels[..., 2:3]  # anchor_x + right
        y2 = anchors[..., 1:2] + dist_pixels[..., 3:4]  # anchor_y + bottom
        
        boxes = torch.cat([x1, y1, x2, y2], -1)  # [B, A, 4]
        
        # Clamp to valid image range
        boxes = boxes.clamp(0, 640)
        
        return boxes

    def forward(self, preds, targets):
        """
        Calculate loss
        
        Args:
            preds: list of 3 tensors [P3, P4, P5] from model
            targets: [N, 6] tensor [batch_idx, class, x_norm, y_norm, w_norm, h_norm]
        """
        cls_out, reg_out = [], []

        for p in preds:
            d, c = p.split((self.reg_max * 4, self.nc), 1)
            cls_out.append(c.flatten(2).permute(0, 2, 1))
            reg_out.append(d.flatten(2).permute(0, 2, 1))

        pred_scores = torch.cat(cls_out, 1)  # [B, 8400, 80]
        pred_dist = torch.cat(reg_out, 1)    # [B, 8400, 64]
        pred_boxes = self.decode(pred_dist)  # [B, 8400, 4]

        B = pred_scores.shape[0]
        
        gt_labels, gt_boxes, mask_gt = self.build_targets(targets, B)

        t_labels, t_boxes, t_scores, fg_mask = self.assigner.forward(
            pred_scores.sigmoid().detach(), 
            pred_boxes.detach(), 
            self.anchors,
            gt_labels, 
            gt_boxes, 
            mask_gt
        )

        score_sum = max(t_scores.sum().item(), 1.0)
        
        # Classification loss
        loss_cls = self.bce(pred_scores, t_scores).sum() / score_sum
        
        # Box loss
        loss_box, loss_dfl = self.bbox_loss(
            pred_dist, pred_boxes, self.anchors, self.strides,
            t_boxes, t_scores, score_sum, fg_mask
        )

        # Apply weights
        loss_box = loss_box * self.box_weight
        loss_cls = loss_cls * self.cls_weight
        loss_dfl = loss_dfl * self.dfl_weight
        
        total_loss = loss_box + loss_cls + loss_dfl

        return {
            "box": loss_box,
            "cls": loss_cls,
            "dfl": loss_dfl,
            "total": total_loss
        }

    def build_targets(self, targets, B):
        """Build ground truth targets from normalized labels"""
        device = targets.device
        
        # Count max GT per image
        max_gt = 0
        for i in range(B):
            n = (targets[:, 0] == i).sum().item()
            max_gt = max(max_gt, n)
        
        if max_gt == 0:
            max_gt = 1
        
        # Initialize
        labels = torch.full((B, max_gt), -1, dtype=torch.long, device=device)
        boxes = torch.zeros((B, max_gt, 4), device=device)
        mask = torch.zeros((B, max_gt), dtype=torch.bool, device=device)

        img_size = 640.0
        
        for i in range(B):
            t = targets[targets[:, 0] == i]
            if len(t) == 0:
                continue

            n = len(t)
            labels[i, :n] = t[:, 1].long()
            
            # Normalized coords to pixels
            x_norm, y_norm = t[:, 2], t[:, 3]
            w_norm, h_norm = t[:, 4], t[:, 5]
            
            x_center = x_norm * img_size
            y_center = y_norm * img_size
            w = w_norm * img_size
            h = h_norm * img_size
            
            # Convert to xyxy
            boxes[i, :n, 0] = x_center - w / 2
            boxes[i, :n, 1] = y_center - h / 2
            boxes[i, :n, 2] = x_center + w / 2
            boxes[i, :n, 3] = y_center + h / 2
            
            boxes[i, :n] = boxes[i, :n].clamp(0, img_size - 1e-3)
            mask[i, :n] = True

        return labels, boxes, mask
