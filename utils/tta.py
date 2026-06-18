"""
Test Time Augmentation (TTA) for Object Detection.
Supports:
  - Horizontal flip: infers model on horizontally flipped image, flips bboxes back, and merges.
  - Multi-scale inference: runs model across multiple image resolutions, rescales outputs, and merges.
  - Combined Mode: combines horizontal flip with multi-scale inputs to yield 2xN model forward passes.
"""
import torch
import torch.nn.functional as F
from utils.nms import non_max_suppression


def _decode_preds(preds, img_size):
    """Decode raw model prediction tensors into bounding box coordinate spaces."""
    from models.loss import decode_predictions_fcos, STRIDES
    return [decode_predictions_fcos(c, r, ct, s, img_size)
            for (c, r, ct), s in zip(preds, STRIDES)]


def _flip_boxes(decoded_tensor, img_size):
    """Flip decoded bounding boxes horizontally (x1_new = img_size - x2_old)."""
    d = decoded_tensor.clone()
    x1 = d[:, 0].clone()
    x2 = d[:, 2].clone()
    d[:, 0] = img_size - x2
    d[:, 2] = img_size - x1
    return d


def _scale_boxes(decoded_tensor, src_size, dst_size):
    """Rescale decoded bounding box coordinates from a source spatial size to a destination spatial size."""
    d = decoded_tensor.clone()
    ratio = dst_size / src_size
    d[:, :4] = (d[:, :4] * ratio).clamp(0, dst_size)
    return d


def tta_forward(model, images, img_size=640,
                flip=True, scales=None,
                conf_thresh=0.01, iou_thresh=0.45,
                soft_nms=False):
    """
    Perform Test Time Augmentation (TTA) using scale variations and horizontal flips.
    Merges multi-augmented predictions before executing Non-Maximum Suppression.

    Args:
        model: neural network model in evaluation mode.
        images: input image batch tensor of shape [B, 3, H, W].
        img_size: default model resolution width/height.
        flip: boolean flag enabling horizontal flip TTA.
        scales: list of extra resolutions to evaluate.
        conf_thresh: classification confidence threshold limit.
        iou_thresh: intersection over union threshold for NMS.
        soft_nms: boolean flag enabling soft NMS.

    Returns:
        list containing lists of prediction dictionary detections per image.
    """
    B = images.shape[0]
    all_decoded = [[] for _ in range(B)]

    # 1. Evaluate original images
    preds = model(images)
    decoded_scales = _decode_preds(preds, img_size)
    for b in range(B):
        for d in decoded_scales:
            all_decoded[b].append(d[b])

    # 2. Evaluate horizontally flipped versions
    if flip:
        flipped = torch.flip(images, dims=[3])  # Flip across spatial width axis
        preds_flip = model(flipped)
        decoded_flip = _decode_preds(preds_flip, img_size)
        for b in range(B):
            for d in decoded_flip:
                all_decoded[b].append(_flip_boxes(d[b], img_size))

    # 3. Evaluate multi-scale versions
    if scales:
        for scale in scales:
            # Interpolate normalized image tensor to target resolution
            scaled_images = F.interpolate(
                images, size=(scale, scale),
                mode='bilinear', align_corners=False
            )
            preds_s = model(scaled_images)
            decoded_s = _decode_preds(preds_s, scale)
            for b in range(B):
                for d in decoded_s:
                    all_decoded[b].append(_scale_boxes(d[b], scale, img_size))

            # Joint horizontal flip and scale evaluation
            if flip:
                flipped_s = torch.flip(scaled_images, dims=[3])
                preds_fs = model(flipped_s)
                decoded_fs = _decode_preds(preds_fs, scale)
                for b in range(B):
                    for d in decoded_fs:
                        flipped_d = _flip_boxes(d[b], scale)
                        all_decoded[b].append(_scale_boxes(flipped_d, scale, img_size))

    # 4. Execute NMS across all compiled prediction views
    results = []
    for b in range(B):
        results.append(non_max_suppression(
            all_decoded[b], conf_thresh, iou_thresh, soft_nms=soft_nms
        ))

    return results
