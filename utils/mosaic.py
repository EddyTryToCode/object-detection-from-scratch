"""
Mosaic Data Augmentation Implementation.
Composes four images into a single training image to increase spatial diversity.
"""
import numpy as np
import random


def mosaic_augment(images, boxes_list, labels_list, img_size=640):
    """
    Combine four images (HWC format) into a single mosaic image.

    Args:
        images: list of four images as numpy arrays of shape [H, W, C].
        boxes_list: list of four bounding box arrays of shape [N, 4] (Pascal VOC format: x1, y1, x2, y2).
        labels_list: list of four class label arrays.
        img_size: targeted output size for the mosaic canvas.

    Returns:
        tuple containing:
            - mosaic_img: composed image array of shape [img_size, img_size, C].
            - mosaic_boxes: mapped bounding boxes array of shape [M, 4].
            - mosaic_labels: mapped labels array of shape [M].
    """
    s = img_size

    # Random intersection point coordinates (center of the mosaic grid)
    yc = int(random.uniform(s * 0.25, s * 0.75))
    xc = int(random.uniform(s * 0.25, s * 0.75))

    mosaic_img = np.zeros((s, s, 3), dtype=np.uint8)
    all_boxes = []
    all_labels = []

    # Placements and coordinates on the canvas for each of the four quadrants
    placements = [
        # (y_start, y_end, x_start, x_end) on canvas
        (0,  yc, 0,  xc),   # top-left
        (0,  yc, xc, s),    # top-right
        (yc, s,  0,  xc),   # bottom-left
        (yc, s,  xc, s),    # bottom-right
    ]

    for i in range(4):
        img = images[i]
        h, w = img.shape[:2]
        boxes = boxes_list[i].copy() if len(boxes_list[i]) > 0 else np.zeros((0, 4), dtype=np.float32)
        labels = labels_list[i].copy() if len(labels_list[i]) > 0 else np.zeros((0,), dtype=np.int64)

        y1_c, y2_c, x1_c, x2_c = placements[i]
        place_h = y2_c - y1_c
        place_w = x2_c - x1_c

        # Scale original image to fit the placement area
        scale = min(place_h / h, place_w / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize image
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img)
        pil_img = pil_img.resize((new_w, new_h), PILImage.BILINEAR)
        resized = np.array(pil_img)

        # Place resized image in canvas (align with top-left of placement coordinates)
        paste_y1 = y1_c
        paste_x1 = x1_c
        paste_y2 = min(paste_y1 + new_h, s)
        paste_x2 = min(paste_x1 + new_w, s)
        actual_h = paste_y2 - paste_y1
        actual_w = paste_x2 - paste_x1

        mosaic_img[paste_y1:paste_y2, paste_x1:paste_x2] = resized[:actual_h, :actual_w]

        # Transform bounding boxes
        if len(boxes) > 0:
            # Scale boxes coordinate values
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + paste_x1
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + paste_y1

            # Clip boxes to canvas boundaries
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, s)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, s)

            # Filter out boxes that became too small after clipping (keep if area > 20% of original scaled area)
            orig_areas = (boxes_list[i][:, 2] - boxes_list[i][:, 0]) * \
                         (boxes_list[i][:, 3] - boxes_list[i][:, 1]) * scale * scale
            new_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            valid = new_areas > orig_areas * 0.2
            # Enforce minimum size threshold constraints
            valid &= (boxes[:, 2] - boxes[:, 0]) > 2
            valid &= (boxes[:, 3] - boxes[:, 1]) > 2

            if valid.any():
                all_boxes.append(boxes[valid])
                all_labels.append(labels[valid])

    if len(all_boxes) > 0:
        mosaic_boxes = np.concatenate(all_boxes, axis=0).astype(np.float32)
        mosaic_labels = np.concatenate(all_labels, axis=0).astype(np.int64)
    else:
        mosaic_boxes = np.zeros((0, 4), dtype=np.float32)
        mosaic_labels = np.zeros((0,), dtype=np.int64)

    return mosaic_img, mosaic_boxes, mosaic_labels


def mixup(img1, boxes1, labels1, img2, boxes2, labels2):
    """
    Perform MixUp augmentation by taking a weighted combination of two images.
    MixUp ratio is sampled from a symmetric Beta distribution.
    """
    r = np.random.beta(32.0, 32.0)  # Thường xoay quanh 0.5
    mix_img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
    
    if len(boxes1) > 0 and len(boxes2) > 0:
        mix_boxes = np.concatenate((boxes1, boxes2), axis=0)
        mix_labels = np.concatenate((labels1, labels2), axis=0)
    elif len(boxes1) > 0:
        mix_boxes = boxes1
        mix_labels = labels1
    elif len(boxes2) > 0:
        mix_boxes = boxes2
        mix_labels = labels2
    else:
        mix_boxes = np.zeros((0, 4), dtype=np.float32)
        mix_labels = np.zeros((0,), dtype=np.int64)
        
    return mix_img, mix_boxes, mix_labels
