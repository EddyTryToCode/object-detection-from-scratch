"""
Training script for FCOS object detection.

Usage:
    python train.py
"""
import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from utils.dataset import DetectionDataset, collate_fn
from utils.augmentation import get_train_transform, get_val_transform, get_mosaic_post_transform
from models.detector import Detector
from utils.nms import non_max_suppression, rescale_boxes
from utils.metrics import compute_map


def parse_args():
    parser = argparse.ArgumentParser(description="Train FCOS object detector")
    parser.add_argument('--train_data',     default='./public/annotations/train.json')
    parser.add_argument('--val_data',       default='./public/annotations/val.json')
    parser.add_argument('--image_dir',      default='./public/train/images')
    parser.add_argument('--val_image_dir',  default='./public/val/images')
    parser.add_argument('--checkpoint_dir', default='./models/')
    return parser.parse_args()


def get_loss_fn():
    from models.loss import compute_loss
    return compute_loss


def decode_and_nms(preds, img_size, conf_thresh, iou_thresh, soft_nms=False):
    """Decode raw FCOS predictions → list of decoded tensors per image → NMS."""
    from models.loss import decode_predictions_fcos, STRIDES
    decoded = []
    for (cls_l, reg_p, ctr_p), stride in zip(preds, STRIDES):
        decoded.append(decode_predictions_fcos(cls_l, reg_p, ctr_p, stride, img_size))
    B = decoded[0].shape[0]
    results = []
    for b in range(B):
        scales = [d[b] for d in decoded]
        results.append(non_max_suppression(scales, conf_thresh, iou_thresh, soft_nms=soft_nms))
    return results


def build_gt_annotations(val_data_path):
    gt = defaultdict(list)
    with open(val_data_path, 'r') as f:
        data = json.load(f)
    for ann in data.get('annotations', []):
        gt[ann['image_id']].append({'class': ann['class'], 'bbox': ann['bbox']})
    return gt


import math
import copy

class ModelEMA:
    """Model Exponential Moving Average for stable validation."""
    def __init__(self, model, decay=0.9998):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / 2000))
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.copy_(v * d + (1. - d) * msd[k])


@torch.no_grad()
def evaluate_val(model, val_loader, device, args, gt_annotations):
    model.eval()
    all_preds = []
    val_conf = 0.05

    for images, targets in tqdm(val_loader, desc="  Validation", leave=False):
        images = images.to(device, non_blocking=True)
        preds_raw = model(images)

        batch_results = decode_and_nms(preds_raw, args.img_size, val_conf, args.iou_thresh,
                                        soft_nms=getattr(args, 'soft_nms', False))

        for b, boxes in enumerate(batch_results):
            orig_size = targets[b]['orig_size']
            image_id = targets[b]['image_id']
            boxes = rescale_boxes(boxes, orig_size, args.img_size)
            for box in boxes:
                all_preds.append({
                    'image_id': image_id, 'class': box['class'],
                    'confidence': box['confidence'], 'bbox': box['bbox'],
                })

    model.train()
    return compute_map(all_preds, gt_annotations)


def main():
    args = parse_args()

    # Training configuration parameters
    args.arch = 'fcos'
    args.epochs = 80
    args.batch_size = 16
    args.lr = 1e-3
    args.img_size = 640
    args.resume = None
    args.num_workers = 4
    args.val_interval = 5
    args.conf_thresh = 0.3
    args.iou_thresh = 0.5
    args.soft_nms = False
    args.patience = 10
    
    # FCOS specific architectural configurations
    args.use_sppf = True
    args.attention = 'cbam'
    args.use_dcn = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"╔══════════════════════════════════════════╗")
    print(f"║  Architecture: {args.arch:>25s} ║")
    print(f"║  Device:       {str(device):>25s} ║")
    if device.type == 'cuda':
        print(f"║  GPU:          {torch.cuda.get_device_name(0):>25s} ║")
    print(f"╚══════════════════════════════════════════╝")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize datasets
    train_dataset = DetectionDataset(args.train_data, args.image_dir,
                                      get_train_transform(args.img_size, args.img_size))
    # Base dataset without transformations, used for Mosaic generator
    raw_dataset = DetectionDataset(args.train_data, args.image_dir, transform=None)
    # Wrap the training dataset with MosaicDataset wrapper
    from utils.mosaic_dataset import MosaicDataset
    mosaic_train = MosaicDataset(
        base_dataset=train_dataset,
        raw_dataset=raw_dataset,
        post_transform=get_mosaic_post_transform(args.img_size, args.img_size),
        img_size=args.img_size,
        p=0.5  # 50% Mosaic, 50% augmentation thường
    )
    val_dataset = DetectionDataset(args.val_data, args.val_image_dir,
                                    get_val_transform(args.img_size, args.img_size))
    train_loader = DataLoader(mosaic_train, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=args.num_workers,
                               pin_memory=True, persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers,
                             pin_memory=True, persistent_workers=(args.num_workers > 0))

    print(f"Train: {len(train_dataset)} ảnh | Val: {len(val_dataset)} ảnh")

    # Map attention string to model configuration
    attn = None if args.attention == 'none' else args.attention

    # Initialize detector model
    model = Detector(
        arch=args.arch, num_classes=5, pretrained_backbone=True,
        use_sppf=args.use_sppf,
        attention=attn,
        use_dcn=args.use_dcn,
    ).to(device)
    loss_fn = get_loss_fn()

    # Optimizer: lower learning rate (0.05x) for pretrained backbone to keep weights stable
    bb = list(model.backbone.parameters())
    other = [p for p in model.parameters() if not any(p is bp for bp in bb)]
    optimizer = torch.optim.AdamW([
        {'params': bb, 'lr': args.lr * 0.05},
        {'params': other, 'lr': args.lr},
    ], weight_decay=1e-4)
    # Cosine annealing scheduler: reach minimum learning rate early for stability
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - 5, 1), eta_min=1e-6
    )

    start_epoch, best_map = 0, 0.0
    ema = ModelEMA(model)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'ema_state_dict' in ckpt:
            ema.ema.load_state_dict(ckpt['ema_state_dict'])
            ema.updates = ckpt.get('ema_updates', 0)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_map = ckpt.get('best_map', 0.0)
        print(f"Resumed from epoch {start_epoch}, best_map={best_map:.4f}")

    gt_annotations = build_gt_annotations(args.val_data)

    print(f"Bắt đầu huấn luyện [{args.arch}]...")
    no_improve_count = 0  # Early stopping counter initialization
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if epoch < 5:
            wf = (epoch + 1) / 5.0
            for i, pg in enumerate(optimizer.param_groups):
                pg['lr'] = (args.lr * 0.05 if i == 0 else args.lr) * wf

        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs}")
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = [{'boxes': t['boxes'].to(device, non_blocking=True),
                         'labels': t['labels'].to(device, non_blocking=True),
                         'image_id': t['image_id'], 'orig_size': t['orig_size']}
                        for t in targets]

            optimizer.zero_grad(set_to_none=True)
            
            # Use bfloat16 mixed-precision to avoid numerical overflow of activation functions (e.g. SiLU)
            # and to significantly reduce memory footprint to mitigate Out-Of-Memory (OOM) issues.
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                preds = model(images)
                loss, ld = loss_fn(preds, targets, img_size=args.img_size)

            # Skip step if NaN or infinite losses are encountered
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ⚠ NaN/Inf detected, skipping batch")
                continue

            # GradScaler is not required for bfloat16 since its dynamic range matches float32
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            # Update Exponential Moving Average (EMA) model weights at each iteration
            ema.update(model)

            epoch_loss += ld['loss']
            pbar.set_postfix({k: f"{v:.3f}" if isinstance(v, float) else v for k, v in ld.items()})

        if epoch >= 5:
            scheduler.step()

        n = len(train_loader)
        print(f"  Epoch {epoch+1:02d} — avg_loss={epoch_loss/n:.4f}")

        ckpt = {'epoch': epoch, 'arch': args.arch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.ema.state_dict(),
                'ema_updates': ema.updates,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_map': best_map, 'config': vars(args)}
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'last.pth'))

        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            # Perform validation using EMA weights
            val_map = evaluate_val(ema.ema, val_loader, device, args, gt_annotations)
            print(f"  [Val] mAP@0.5 = {val_map:.4f}  (best = {best_map:.4f})")
            if val_map > best_map:
                best_map = val_map
                ckpt['best_map'] = best_map
                torch.save(ckpt, os.path.join(args.checkpoint_dir, 'best.pth'))
                print(f"  ✓ Saved best.pth (mAP={best_map:.4f})")
                no_improve_count = 0
            else:
                no_improve_count += args.val_interval
                if no_improve_count >= args.patience:
                    print(f"  ✗ Early stopping: {no_improve_count} epochs without improvement")
                    break

    print(f"\nDone [{args.arch}]! Best mAP@0.5 = {best_map:.4f}")


if __name__ == "__main__":
    main()
