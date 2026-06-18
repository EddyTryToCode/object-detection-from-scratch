"""
Evaluation metrics helper functions.
Includes intersection-over-union (IoU) utilities and mAP@0.5 computation functions.
"""
from collections import defaultdict


IDX_TO_CLASS = {0: "person", 1: "car", 2: "dog", 3: "cat", 4: "chair"}
CLASSES = ["person", "car", "dog", "cat", "chair"]


def compute_iou(box1, box2):
    """Calculate IoU between two boxes format [x1, y1, x2, y2]."""
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / (union_area + 1e-6)


def compute_ap(recalls, precisions):
    """Compute Average Precision (AP) using standard 11-point interpolation."""
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def compute_map(pred_results, gt_annotations, iou_threshold=0.5):
    """
    Compute mean Average Precision at IoU threshold of 0.5 (mAP@0.5).

    Args:
        pred_results: list of dict containing prediction mappings.
        gt_annotations: dict mapping image IDs to their corresponding ground truth labels.

    Returns:
        float value representing mAP@0.5.
    """
    # Group GT by class
    gt_by_class = defaultdict(lambda: defaultdict(list))
    for img_id, anns in gt_annotations.items():
        for ann in anns:
            gt_by_class[ann["class"]][img_id].append({
                "bbox": ann["bbox"],
                "matched": False
            })

    # Group predictions by class
    pred_by_class = defaultdict(list)
    for pred in pred_results:
        pred_by_class[pred["class"]].append(pred)

    aps = []
    for cls in CLASSES:
        cls_gt = gt_by_class[cls]
        num_gt = sum(len(v) for v in cls_gt.values())
        if num_gt == 0:
            continue

        cls_preds = sorted(pred_by_class[cls], key=lambda x: x["confidence"], reverse=True)
        tp_flags, fp_flags = [], []

        for pred in cls_preds:
            candidates = cls_gt.get(pred["image_id"], [])
            best_iou, best_idx = 0.0, -1
            for idx, gt in enumerate(candidates):
                if gt["matched"]:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_idx >= 0 and best_iou >= iou_threshold:
                candidates[best_idx]["matched"] = True
                tp_flags.append(1); fp_flags.append(0)
            else:
                tp_flags.append(0); fp_flags.append(1)

        cum_tp, cum_fp, tp_sum, fp_sum = [], [], 0, 0
        for tp, fp in zip(tp_flags, fp_flags):
            tp_sum += tp; fp_sum += fp
            cum_tp.append(tp_sum); cum_fp.append(fp_sum)

        recalls    = [t / num_gt for t in cum_tp]
        precisions = [t / max(t + f, 1) for t, f in zip(cum_tp, cum_fp)]
        aps.append(compute_ap(recalls, precisions))

    return sum(aps) / len(aps) if aps else 0.0
