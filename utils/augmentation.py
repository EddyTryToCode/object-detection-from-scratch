"""
Data Augmentation Pipelines for Object Detection.
Includes standard geometric and color-space augmentations.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(height=640, width=640):
    return A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(size=(height, width), scale=(0.3, 1.0), ratio=(0.5, 2.0), p=0.7),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5,
                           border_mode=0),  # cv2.BORDER_CONSTANT

        # Color-space transformations
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15, p=0.6),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.ToGray(p=0.05),  # Random conversion to grayscale to encourage shape learning

        # Final resize + normalize
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels'],
        min_visibility=0.2,
        clip=True
    ))


def get_mosaic_post_transform(height=640, width=640):
    """Post-processing transformations applied after Mosaic image composition.
    Geometric augmentations are omitted here since Mosaic provides spatial diversity.
    """
    return A.Compose([
        # Color augmentations
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15, p=0.6),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.2),

        # Final resize + normalize
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels'],
        min_visibility=0.2,
        clip=True
    ))


def get_val_transform(height=640, width=640):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels'],
        clip=True
    ))
