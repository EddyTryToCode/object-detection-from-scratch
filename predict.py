"""
FCOS Object Detection Inference Script.

Usage:
    python predict.py --image_dir ./public/val/images --output predictions.json
"""
import os
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

from utils.augmentation import get_val_transform
from models.detector import Detector
from utils.nms import non_max_suppression, rescale_boxes

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


class InferenceDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_paths = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        orig_w, orig_h = img.size
        img_np = np.array(img)
        transformed = self.transform(image=img_np, bboxes=[], class_labels=[])
        return transformed['image'], path.name, (orig_h, orig_w)


def collate_inference(batch):
    images = torch.stack([b[0] for b in batch])
    filenames = [b[1] for b in batch]
    orig_sizes = [b[2] for b in batch]
    return images, filenames, orig_sizes


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for FCOS")
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--output', default='predictions.json')
    return parser.parse_args()


def decode_and_nms(preds, img_size, conf_thresh, iou_thresh):
    from models.loss import decode_predictions_fcos, STRIDES
    decoded = [decode_predictions_fcos(c, r, ct, s, img_size) for (c, r, ct), s in zip(preds, STRIDES)]
    B = decoded[0].shape[0]
    return [non_max_suppression([d[b] for d in decoded], conf_thresh, iou_thresh) for b in range(B)]


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Inference configuration parameters
    weights_path = './models/best.pth'
    
    # Automatically download pre-trained weights from Hugging Face if not found locally
    WEIGHTS_URL = "https://huggingface.co/EddyTryToCode/fcos-object-detection/resolve/main/best.pth"
    
    if not os.path.exists(weights_path):
        print(f"Weights file not found at {weights_path}.")
        print(f"Downloading weights automatically from Hugging Face: {WEIGHTS_URL} ...")
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        try:
            import urllib.request
            urllib.request.urlretrieve(WEIGHTS_URL, weights_path)
            print(">>> Weights downloaded successfully!")
        except Exception as e:
            print(f"Error downloading weights: {e}")
            print("Please verify the Hugging Face URL or manually place the weights file at models/best.pth")
            sys.exit(1)
        
    img_size = 640
    batch_size = 8
    num_workers = 4
    conf_thresh = 0.001
    iou_thresh = 0.65
    use_ema = True
    soft_nms = False
    use_tta = True
    tta_scales = [480, 576, 704, 832]

    # FCOS architectural configurations
    use_sppf = True
    attention = 'cbam'
    use_dcn = True

    print(f"Arch: fcos | Weights: {weights_path} | Device: {device}")

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = Detector(
        num_classes=5, pretrained_backbone=False,
        use_sppf=use_sppf, attention=attention, use_dcn=use_dcn
    ).to(device)

    # Load EMA weights if available and requested
    if use_ema and 'ema_state_dict' in ckpt:
        model.load_state_dict(ckpt['ema_state_dict'])
        print("  Loaded EMA weights ✓")
    else:
        model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    transform = get_val_transform(img_size, img_size)
    dataset = InferenceDataset(args.image_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, collate_fn=collate_inference, pin_memory=True)
    print(f"Images: {len(dataset)}")

    results_map = {}

    if use_tta:
        from utils.tta import tta_forward
        print(f"TTA enabled: flip=True, scales={tta_scales}")

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        for images, filenames, orig_sizes in tqdm(loader, desc="Predicting"):
            images = images.to(device, non_blocking=True)

            if use_tta:
                batch_results = tta_forward(
                    model, images, img_size=img_size,
                    flip=True, scales=tta_scales,
                    conf_thresh=conf_thresh, iou_thresh=iou_thresh,
                    soft_nms=soft_nms
                )
            else:
                preds = model(images)
                batch_results = decode_and_nms(
                    preds, img_size, conf_thresh, iou_thresh
                )

            for b, boxes in enumerate(batch_results):
                boxes = rescale_boxes(boxes, orig_sizes[b], img_size)
                results_map[filenames[b]] = [
                    {"class": bx["class"], "confidence": bx["confidence"], "bbox": bx["bbox"]}
                    for bx in boxes
                ]

    # Ensure all dataset images have an entry in the output map
    for p in dataset.image_paths:
        results_map.setdefault(p.name, [])

    output_list = [{"image_id": k, "boxes": v} for k, v in sorted(results_map.items())]
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_list, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(output_list)} results → {args.output}")


if __name__ == "__main__":
    main()
