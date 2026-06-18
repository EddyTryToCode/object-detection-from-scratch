"""
MosaicDataset wrapper class.
Wraps the base DetectionDataset to apply Mosaic and optional MixUp data augmentations during training.
"""
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.mosaic import mosaic_augment, mixup


class MosaicDataset(Dataset):
    """
    Dataset wrapper to dynamically apply Mosaic data augmentation.

    Args:
        base_dataset: DetectionDataset instance configured with standard augmentations.
        raw_dataset: DetectionDataset instance configured without augmentations (used for retrieving raw images).
        post_transform: Albumentations transformations applied after the Mosaic composition.
        img_size: Target image size for training.
        p: Probability of applying Mosaic augmentation.
    """

    def __init__(self, base_dataset, raw_dataset, post_transform=None, img_size=640, p=0.5, p_mixup=0.15):
        self.base_dataset = base_dataset
        self.raw_dataset = raw_dataset
        self.post_transform = post_transform
        self.img_size = img_size
        self.p = p
        self.p_mixup = p_mixup

    def __len__(self):
        return len(self.base_dataset)

    def _get_raw_item(self, idx):
        """Retrieve the raw image array and its ground truth annotations."""
        img_id = self.raw_dataset.image_ids[idx]
        img_info = self.raw_dataset.images_info[img_id]

        import os
        from PIL import Image
        img_name = os.path.basename(img_info['file_name'])
        img_path = os.path.join(self.raw_dataset.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        image_np = np.array(img)

        anns = self.raw_dataset.annotations[img_id]
        boxes = []
        labels = []
        for ann in anns:
            boxes.append(ann['bbox'])
            labels.append(self.raw_dataset.class_to_idx[ann['class']])

        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        return image_np, boxes, labels, img_id, (orig_h, orig_w)

    def _build_mosaic(self, idx):
        indices = [idx] + [random.randint(0, len(self.raw_dataset) - 1) for _ in range(3)]
        images, boxes_list, labels_list = [], [], []
        first_img_id = None
        first_orig_size = None

        for i, j in enumerate(indices):
            img_np, boxes, labels, img_id, orig_size = self._get_raw_item(j)
            images.append(img_np)
            boxes_list.append(boxes)
            labels_list.append(labels)
            if i == 0:
                first_img_id = img_id
                first_orig_size = orig_size

        mosaic_img, mosaic_boxes, mosaic_labels = mosaic_augment(
            images, boxes_list, labels_list, self.img_size
        )
        return mosaic_img, mosaic_boxes, mosaic_labels, first_img_id, first_orig_size

    def __getitem__(self, idx):
        if random.random() < self.p:
            # 1. Generate the base Mosaic image composition
            mosaic_img, mosaic_boxes, mosaic_labels, first_img_id, first_orig_size = self._build_mosaic(idx)

            # 2. Optionally apply MixUp augmentation with probability p_mixup
            if random.random() < self.p_mixup:
                idx2 = random.randint(0, len(self.raw_dataset) - 1)
                mosaic_img2, mosaic_boxes2, mosaic_labels2, _, _ = self._build_mosaic(idx2)
                mosaic_img, mosaic_boxes, mosaic_labels = mixup(
                    mosaic_img, mosaic_boxes, mosaic_labels,
                    mosaic_img2, mosaic_boxes2, mosaic_labels2
                )

            # Apply post-augmentation transforms (normalization, tensor conversion)
            if self.post_transform is not None:
                transformed = self.post_transform(
                    image=mosaic_img,
                    bboxes=mosaic_boxes.tolist() if len(mosaic_boxes) > 0 else [],
                    class_labels=mosaic_labels.tolist() if len(mosaic_labels) > 0 else []
                )
                image_tensor = transformed['image']
                t_boxes = transformed['bboxes']
                t_labels = transformed['class_labels']

                if len(t_boxes) > 0:
                    boxes_out = torch.tensor(t_boxes, dtype=torch.float32)
                    labels_out = torch.tensor(t_labels, dtype=torch.long)
                else:
                    boxes_out = torch.zeros((0, 4), dtype=torch.float32)
                    labels_out = torch.zeros((0,), dtype=torch.long)
            else:
                # Fallback mechanism when post_transform is not provided
                import torchvision.transforms.functional as TF
                image_tensor = TF.to_tensor(mosaic_img)
                boxes_out = torch.tensor(mosaic_boxes, dtype=torch.float32)
                labels_out = torch.tensor(mosaic_labels, dtype=torch.long)

            target = {
                "boxes": boxes_out,
                "labels": labels_out,
                "image_id": first_img_id,
                "orig_size": first_orig_size,
            }
            return image_tensor, target

        else:
            # Bypass Mosaic: retrieve item from standard base dataset directly
            return self.base_dataset[idx]
