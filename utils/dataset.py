"""
Dataset class definition for object detection.
Loads image files, parses annotations, and formats targets for the FCOS detector.
"""
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class DetectionDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        self.class_to_idx = {"person": 0, "car": 1, "dog": 2, "cat": 3, "chair": 4}
        
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.images_info = {img['id']: img for img in data['images']}
        self.image_ids = list(self.images_info.keys())
        
        # Build dict mapping image_id to list of annotations
        self.annotations = {img_id: [] for img_id in self.image_ids}
        for ann in data.get('annotations', []):
            self.annotations[ann['image_id']].append(ann)
            
    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images_info[img_id]
        
        # Load image
        img_name = os.path.basename(img_info['file_name'])
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        
        # Get annotations
        anns = self.annotations[img_id]
        boxes = []
        labels = []
        
        for ann in anns:
            boxes.append(ann['bbox'])
            labels.append(self.class_to_idx[ann['class']])
            
        # Convert to numpy array for Albumentations
        image_np = np.array(img)
        
        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            
        # Apply transforms if provided
        if self.transform is not None:
            # Albumentations signature
            transformed = self.transform(image=image_np, bboxes=boxes, class_labels=labels)
            image_tensor = transformed['image']
            
            t_boxes = transformed['bboxes']
            t_labels = transformed['class_labels']
            
            if len(t_boxes) > 0:
                boxes = torch.tensor(t_boxes, dtype=torch.float32)
                labels = torch.tensor(t_labels, dtype=torch.long)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.long)
        else:
            # Fallback if no transform is provided
            import torchvision.transforms.functional as TF
            image_tensor = TF.to_tensor(img)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id,
            "orig_size": (orig_h, orig_w)
        }
        
        return image_tensor, target

def collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # Stack images into shape [B, C, H, W]
    images = torch.stack(images, dim=0)
    return images, targets
