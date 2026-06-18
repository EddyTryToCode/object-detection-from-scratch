import numpy as np
import torch


IDX_TO_CLASS = {0: "person", 1: "car", 2: "dog", 3: "cat", 4: "chair"}
NUM_CLASSES = 5


def _compute_iou_matrix(boxes):
    """
    Compute pairwise IoU matrix for a set of bounding boxes (vectorized).

    Args:
        boxes: numpy array of shape [N, 4] representing coordinates (x1, y1, x2, y2).

    Returns:
        numpy array of shape [N, N] containing pairwise IoU values.
    """
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)  # Box areas of shape [N]

    # Broadcast box coordinates to compute intersection area of all pairs
    inter_x1 = np.maximum(x1[:, None], x1[None, :])
    inter_y1 = np.maximum(y1[:, None], y1[None, :])
    inter_x2 = np.minimum(x2[:, None], x2[None, :])
    inter_y2 = np.minimum(y2[:, None], y2[None, :])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h  # [N, N]

    union_area = areas[:, None] + areas[None, :] - inter_area
    iou = inter_area / np.maximum(union_area, 1e-6)
    return iou


def compute_iou(box1, box2):
    """Compute IoU between two individual bounding boxes with format [x1, y1, x2, y2]."""
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / (union_area + 1e-6)


def non_max_suppression(decoded_scales, conf_threshold=0.3, iou_threshold=0.45,
                        soft_nms=False, soft_nms_sigma=0.5):
    """
    Perform per-class Non-Maximum Suppression (NMS) in a vectorized fashion.
    Supports standard Hard NMS and Gaussian Soft-NMS.

    Args:
        decoded_scales: list of Tensors of shape [M, 5+C] containing decoded detections.
                        Tensor layout: [x1, y1, x2, y2, obj_conf, class_scores].
        conf_threshold: confidence threshold filters.
        iou_threshold: intersection over union threshold for hard suppression.
        soft_nms: whether to apply Soft-NMS instead of Hard NMS.
        soft_nms_sigma: Gaussian decay standard deviation parameter for Soft-NMS.

    Returns:
        list of dict: filtered detection list containing class names, confidence scores, and bounding boxes.
    """
    # Step 1: Concatenate detections across all scale levels
    all_preds = torch.cat(decoded_scales, dim=0)  # [M, 5+C]

    obj_conf  = all_preds[:, 4]          # [M]
    cls_scores = all_preds[:, 5:]        # [M, C]

    # Step 2: Calculate confidence score = obj_conf * max_class_score
    cls_conf, cls_ids = cls_scores.max(dim=1)   # [M]
    confidence = obj_conf * cls_conf             # [M]

    # Step 3: Filter detections by confidence threshold
    keep = confidence > conf_threshold
    if not keep.any():
        return []

    boxes_t = all_preds[keep, :4]
    confs_t = confidence[keep]
    cls_ids_t = cls_ids[keep]

    # Apply Top-K selection prior to NMS execution to reduce computing overhead
    MAX_PRE_NMS = 3000
    if len(confs_t) > MAX_PRE_NMS:
        _, topk_idx = torch.topk(confs_t, MAX_PRE_NMS)
        boxes_t = boxes_t[topk_idx]
        confs_t = confs_t[topk_idx]
        cls_ids_t = cls_ids_t[topk_idx]

    boxes_np  = boxes_t.cpu().numpy().astype(np.float32)
    confs_np  = confs_t.cpu().numpy().astype(np.float32)
    cls_np    = cls_ids_t.cpu().numpy().astype(np.int32)

    # Step 4: Perform per-class vectorized NMS
    results = []
    for cls_id in range(NUM_CLASSES):
        mask = cls_np == cls_id
        if not mask.any():
            continue

        cls_boxes = boxes_np[mask]   # [N_cls, 4]
        cls_confs = confs_np[mask]   # [N_cls]

        # Sort predictions by confidence scores in descending order
        order = np.argsort(-cls_confs)
        cls_boxes = cls_boxes[order]
        cls_confs = cls_confs[order]

        # Compute pairwise IoU matrix for the active class
        iou_mat = _compute_iou_matrix(cls_boxes)  # [N_cls, N_cls]

        suppressed = np.zeros(len(cls_boxes), dtype=bool)

        if soft_nms:
            # Gaussian Soft-NMS: decay scores instead of hard suppression
            scores = cls_confs.copy()
            for i in range(len(cls_boxes)):
                if scores[i] < conf_threshold:
                    continue
                results.append({
                    "class":      IDX_TO_CLASS[cls_id],
                    "confidence": float(scores[i]),
                    "bbox":       [float(v) for v in cls_boxes[i]],
                })
                # Gaussian decay for remaining boxes
                ious = iou_mat[i, i + 1:]
                scores[i + 1:] *= np.exp(-(ious ** 2) / soft_nms_sigma)
        else:
            # Standard Hard NMS (default)
            for i in range(len(cls_boxes)):
                if suppressed[i]:
                    continue
                results.append({
                    "class":      IDX_TO_CLASS[cls_id],
                    "confidence": float(cls_confs[i]),
                    "bbox":       [float(v) for v in cls_boxes[i]],
                })
                # Suppress remaining candidates with an IoU exceeding the defined threshold
                suppressed[i + 1:] |= (iou_mat[i, i + 1:] > iou_threshold)

    return results


def rescale_boxes(boxes, orig_size, model_size=640):
    """
    Rescale bounding box coordinates from input model dimensions back to original image dimensions.

    Args:
        boxes: list of detection dictionaries.
        orig_size: tuple (orig_H, orig_W) containing the original image spatial size.
        model_size: input resolution of the network.

    Returns:
        list of detection dictionaries with scaled coordinates.
    """
    orig_h, orig_w = orig_size
    scale_x = orig_w / model_size
    scale_y = orig_h / model_size

    rescaled = []
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        x1 = int(round(max(0, min(x1 * scale_x, orig_w))))
        y1 = int(round(max(0, min(y1 * scale_y, orig_h))))
        x2 = int(round(max(0, min(x2 * scale_x, orig_w))))
        y2 = int(round(max(0, min(y2 * scale_y, orig_h))))
        if x2 > x1 and y2 > y1:
            rescaled.append({
                "class":      box["class"],
                "confidence": box["confidence"],
                "bbox":       [x1, y1, x2, y2],
            })

    return rescaled
