"""
Loss functions, Target Assignment, and Prediction Decoding for FCOS.
- Focal Loss for classification tasks.
- CIoU Loss for bounding box regression.
- Binary Cross Entropy (BCE) for centerness estimation.
"""
import math
import torch
import torch.nn.functional as F

STRIDES = [8, 16, 32]
# Scale ranges determining size responsibilities for each FPN level
SCALE_RANGES = [(0, 96), (96, 192), (192, 1e8)]


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, norm=None):
    """Focal Loss for dense classification."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    pt = torch.where(targets == 1, p, 1 - p)
    at = torch.where(targets == 1, alpha, 1 - alpha)
    loss = at * (1 - pt) ** gamma * bce
    if norm is None:
        norm = max(targets.sum(), 1.0)
    return loss.sum() / norm


def ciou(b1, b2):
    """Complete IoU: overlap + center distance + aspect ratio."""
    ix1 = torch.max(b1[:, 0], b2[:, 0]); iy1 = torch.max(b1[:, 1], b2[:, 1])
    ix2 = torch.min(b1[:, 2], b2[:, 2]); iy2 = torch.min(b1[:, 3], b2[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    a1 = ((b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])).clamp(0)
    a2 = ((b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])).clamp(0)
    union = a1 + a2 - inter
    iou = inter / union.clamp(1e-6)
    # Center distance penalty
    pcx = (b1[:, 0] + b1[:, 2]) / 2; pcy = (b1[:, 1] + b1[:, 3]) / 2
    gcx = (b2[:, 0] + b2[:, 2]) / 2; gcy = (b2[:, 1] + b2[:, 3]) / 2
    rho2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2
    # Enclosing box diagonal penalty
    ex1 = torch.min(b1[:, 0], b2[:, 0]); ey1 = torch.min(b1[:, 1], b2[:, 1])
    ex2 = torch.max(b1[:, 2], b2[:, 2]); ey2 = torch.max(b1[:, 3], b2[:, 3])
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-6
    # Aspect ratio consistency term
    w1 = (b1[:, 2] - b1[:, 0]).clamp(1e-6); h1 = (b1[:, 3] - b1[:, 1]).clamp(1e-6)
    w2 = (b2[:, 2] - b2[:, 0]).clamp(1e-6); h2 = (b2[:, 3] - b2[:, 1]).clamp(1e-6)
    v = (4 / (math.pi ** 2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-6)
    return iou - rho2 / c2 - alpha * v


def assign_fcos(gt_boxes, gt_labels, stride, feat_size, scale_range, img_size=640, radius=1.5):
    """
    Assign target labels and bounding boxes to each pixel on the feature map.
    A pixel (x, y) is designated as a positive sample if:
      1. It resides inside a ground-truth (GT) bounding box.
      2. It lies within a defined center sampling radius (radius * stride) of the GT box.
      3. The maximum distance value max(l, t, r, b) matches the FPN level's assigned scale range.
    """
    device = gt_boxes.device
    H, W = feat_size
    lo, hi = scale_range

    # Calculate coordinate values of pixel centers on the input image space
    ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * stride
    ys, xs = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]

    # Initialize output target tensors
    cls_targets = torch.zeros(H, W, dtype=torch.long, device=device)
    reg_targets = torch.zeros(H, W, 4, dtype=torch.float32, device=device)
    ctr_targets = torch.zeros(H, W, dtype=torch.float32, device=device)
    pos_mask = torch.zeros(H, W, dtype=torch.bool, device=device)

    if len(gt_boxes) == 0:
        return cls_targets, reg_targets, ctr_targets, pos_mask

    # Iterate over GT boxes, smallest first so larger boxes can overwrite them in overlaps
    areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    order = areas.argsort()

    for idx in order:
        x1, y1, x2, y2 = gt_boxes[idx]
        label = gt_labels[idx]

        # Compute distance to left (l), top (t), right (r), and bottom (b) box edges
        l = xs - x1; t = ys - y1; r = x2 - xs; b = y2 - ys
        ltrb = torch.stack([l, t, r, b], dim=-1)  # [H, W, 4]
        max_ltrb = ltrb.max(dim=-1).values  # [H, W]

        # Define boundaries for center sampling
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        c_l = xs - (cx - radius * stride)
        c_t = ys - (cy - radius * stride)
        c_r = (cx + radius * stride) - xs
        c_b = (cy + radius * stride) - ys

        # Verify conditions for positive designation
        inside = (l > 0) & (t > 0) & (r > 0) & (b > 0)       # resides inside GT box
        in_center = (c_l > 0) & (c_t > 0) & (c_r > 0) & (c_b > 0) # resides inside center sampling area
        in_range = (max_ltrb >= lo) & (max_ltrb < hi)          # falls within scale range

        valid = inside & in_center & in_range
        if not valid.any():
            continue

        # Calculate centerness targets
        lr = torch.min(l, r) / torch.max(l, r).clamp(1e-6)
        tb = torch.min(t, b) / torch.max(t, b).clamp(1e-6)
        ctr = torch.sqrt((lr * tb).clamp(0))

        cls_targets[valid] = label + 1  # 0: background, class indices shifted to 1..num_classes
        reg_targets[valid] = ltrb[valid]
        ctr_targets[valid] = ctr[valid]
        pos_mask[valid] = True

    return cls_targets, reg_targets, ctr_targets, pos_mask


def compute_loss(predictions, targets, **kwargs):
    """
    predictions: list of (cls_logits, reg_pred, ctr_pred) per scale
    """
    device = predictions[0][0].device
    B = len(targets)
    img_size = kwargs.get('img_size', 640)

    total_cls = torch.zeros(1, device=device)
    total_reg = torch.zeros(1, device=device)
    total_ctr = torch.zeros(1, device=device)
    n_pos = 0
    norm_factor = 0.0

    for scale_idx, (cls_logits, reg_pred, ctr_pred) in enumerate(predictions):
        _, C, H, W = cls_logits.shape
        stride = STRIDES[scale_idx]
        sr = SCALE_RANGES[scale_idx]

        all_cls_t, all_reg_t, all_ctr_t, all_pos = [], [], [], []
        for b in range(B):
            ct, rt, ctrt, pm = assign_fcos(
                targets[b]['boxes'], targets[b]['labels'], stride, (H, W), sr, img_size
            )
            all_cls_t.append(ct); all_reg_t.append(rt)
            all_ctr_t.append(ctrt); all_pos.append(pm)

        cls_t = torch.stack(all_cls_t)   # [B, H, W]
        reg_t = torch.stack(all_reg_t)   # [B, H, W, 4]
        ctr_t = torch.stack(all_ctr_t)   # [B, H, W]
        pos_m = torch.stack(all_pos)     # [B, H, W]
        
        n_pos += pos_m.sum().item()
        # Accumulate centerness values of positive locations to serve as normalization factor
        norm_factor += ctr_t[pos_m].sum().item()

        # Classification branch: convert targets to one-hot vectors for focal loss
        cls_logits_flat = cls_logits.permute(0, 2, 3, 1)  # [B, H, W, C]
        one_hot = torch.zeros_like(cls_logits_flat)
        if pos_m.any():
            pos_labels = cls_t[pos_m] - 1  # Shift back to 0-indexed format
            one_hot_pos = torch.zeros(pos_labels.shape[0], C, device=device, dtype=one_hot.dtype)
            one_hot_pos.scatter_(1, pos_labels.unsqueeze(1), 1)
            one_hot[pos_m] = one_hot_pos
        
        # Accumulate unnormalized classification loss
        total_cls += focal_loss(cls_logits_flat, one_hot, norm=1.0)

        if pos_m.any():
            # Regression branch: CIoU loss computed over decoded bounding boxes
            reg_flat = reg_pred.permute(0, 2, 3, 1)  # [B, H, W, 4]
            pred_ltrb = reg_flat[pos_m]  # [N_pos, 4]
            gt_ltrb   = reg_t[pos_m]     # [N_pos, 4]

            ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * stride
            xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * stride
            ys_g, xs_g = torch.meshgrid(ys, xs, indexing='ij')
            xs_g = xs_g.unsqueeze(0).expand(B, -1, -1)
            ys_g = ys_g.unsqueeze(0).expand(B, -1, -1)

            px = xs_g[pos_m]; py = ys_g[pos_m]
            p_x1 = px - pred_ltrb[:, 0]; p_y1 = py - pred_ltrb[:, 1]
            p_x2 = px + pred_ltrb[:, 2]; p_y2 = py + pred_ltrb[:, 3]
            g_x1 = px - gt_ltrb[:, 0]; g_y1 = py - gt_ltrb[:, 1]
            g_x2 = px + gt_ltrb[:, 2]; g_y2 = py + gt_ltrb[:, 3]

            pbox = torch.stack([p_x1, p_y1, p_x2, p_y2], -1)
            gbox = torch.stack([g_x1, g_y1, g_x2, g_y2], -1)
            ciou_val = ciou(pbox, gbox)  # [N_pos]
            reg_loss = (1 - ciou_val) * ctr_t[pos_m]  # Weighted by centerness targets
            total_reg += reg_loss.sum()

            # Centerness branch
            ctr_flat = ctr_pred.squeeze(1)  # [B, H, W]
            total_ctr += F.binary_cross_entropy_with_logits(
                ctr_flat[pos_m], ctr_t[pos_m], reduction='sum'
            )

    # Normalize losses according to the FCOS framework specification
    norm = max(norm_factor, 1.0)
    total_cls = total_cls / norm
    total_reg = total_reg / norm      # Centerness-weighted normalization
    total_ctr = total_ctr / max(n_pos, 1)  # Normalize by number of positives

    loss = total_cls + total_reg + total_ctr
    return loss, {'loss': loss.item(), 'cls': total_cls.item(),
                   'reg': total_reg.item(), 'ctr': total_ctr.item(), 'pos': n_pos}


def decode_predictions_fcos(cls_logits, reg_pred, ctr_pred, stride, img_size=640):
    """Decode raw model predictions for a single scale level into bounding box coordinates."""
    B, C, H, W = cls_logits.shape
    device = cls_logits.device

    ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * stride
    ys, xs = torch.meshgrid(ys, xs, indexing='ij')
    xs = xs.view(1, H, W).expand(B, -1, -1)
    ys = ys.view(1, H, W).expand(B, -1, -1)

    reg = reg_pred.permute(0, 2, 3, 1)  # [B, H, W, 4] = l,t,r,b
    x1 = (xs - reg[..., 0]).clamp(0, img_size)
    y1 = (ys - reg[..., 1]).clamp(0, img_size)
    x2 = (xs + reg[..., 2]).clamp(0, img_size)
    y2 = (ys + reg[..., 3]).clamp(0, img_size)

    cls_score = torch.sigmoid(cls_logits.permute(0, 2, 3, 1))  # [B,H,W,C]
    ctr_score = torch.sigmoid(ctr_pred.squeeze(1))              # [B,H,W]

    # NMS computes final score by multiplying with class scores; return centerness as raw conf here
    conf = ctr_score.unsqueeze(-1)  # [B,H,W,1]
    out = torch.cat([x1.unsqueeze(-1), y1.unsqueeze(-1),
                     x2.unsqueeze(-1), y2.unsqueeze(-1),
                     conf, cls_score], dim=-1)
    return out.view(B, -1, 5 + C)
