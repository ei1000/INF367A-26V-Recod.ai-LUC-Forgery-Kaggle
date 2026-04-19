from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.ndimage
import scipy.optimize
import torch
import torch.nn.functional as F


def initialize_segmentation_counts() -> dict[str, int]:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "pred_pos": 0,
        "mask_pos": 0,
        "pixels": 0,
    }


def update_segmentation_counts(preds: torch.Tensor, masks: torch.Tensor, counts: dict[str, int]):
    preds_fg = preds == 1
    masks_fg = masks == 1

    counts["tp"] += int((preds_fg & masks_fg).sum().item())
    counts["fp"] += int((preds_fg & ~masks_fg).sum().item())
    counts["fn"] += int((~preds_fg & masks_fg).sum().item())
    counts["pred_pos"] += int(preds_fg.sum().item())
    counts["mask_pos"] += int(masks_fg.sum().item())
    counts["pixels"] += int(masks.numel())


def summarize_segmentation_counts(counts: dict[str, int]):
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    pixels = counts["pixels"]

    iou_den = tp + fp + fn
    dice_den = (2 * tp) + fp + fn

    return {
        "iou": (tp / iou_den) if iou_den else 0.0,
        "dice": ((2 * tp) / dice_den) if dice_den else 0.0,
        "pred_positive_rate": (counts["pred_pos"] / pixels) if pixels else 0.0,
        "mask_positive_rate": (counts["mask_pos"] / pixels) if pixels else 0.0,
    }


def split_mask_instances(mask: np.ndarray) -> list[np.ndarray]:
    if mask.ndim == 2:
        channel_masks = [(mask > 0).astype(np.uint8)]
    elif mask.ndim == 3:
        if mask.shape[0] <= 16 and mask.shape[0] <= mask.shape[-1] and mask.shape[0] <= mask.shape[-2]:
            channel_masks = [(mask[i] > 0).astype(np.uint8) for i in range(mask.shape[0])]
        elif mask.shape[-1] <= 16 and mask.shape[-1] <= mask.shape[0] and mask.shape[-1] <= mask.shape[1]:
            channel_masks = [(mask[..., i] > 0).astype(np.uint8) for i in range(mask.shape[-1])]
        else:
            raise ValueError(f"Could not infer channel axis for mask with shape {mask.shape}")
    else:
        raise ValueError(f"Unsupported mask shape {mask.shape}")

    instances = []
    for channel_mask in channel_masks:
        labeled, count = scipy.ndimage.label(channel_mask)
        for component_idx in range(1, count + 1):
            component = (labeled == component_idx).astype(np.uint8)
            if component.any():
                instances.append(component)
    return instances


def resize_binary_mask(mask: np.ndarray, image_size: int) -> np.ndarray:
    mask_tensor = torch.from_numpy(mask).view(1, 1, *mask.shape).float()
    resized = F.interpolate(mask_tensor, size=(image_size, image_size), mode="nearest")
    return resized.squeeze(0).squeeze(0).numpy().astype(np.uint8)


def load_resized_gt_instances(sample_path: str | Path, mask_dir_by_sample: dict[str | Path, Path | None], image_size: int) -> list[np.ndarray]:
    path = Path(sample_path)
    mask_dir = (
        mask_dir_by_sample.get(sample_path)
        or mask_dir_by_sample.get(str(path))
        or mask_dir_by_sample.get(path)
    )
    if mask_dir is None or "forged" not in path.parent.name:
        return []

    mask_path = mask_dir / path.name.replace(".png", ".npy")
    mask = np.load(mask_path)
    instances = split_mask_instances(mask)
    resized_instances = []
    for instance in instances:
        resized = resize_binary_mask(instance, image_size=image_size)
        if resized.any():
            resized_instances.append(resized)
    return resized_instances


def binary_mask_to_instances(mask: np.ndarray) -> list[np.ndarray]:
    mask = (mask > 0).astype(np.uint8)
    labeled, count = scipy.ndimage.label(mask)
    instances = []
    for component_idx in range(1, count + 1):
        component = (labeled == component_idx).astype(np.uint8)
        if component.any():
            instances.append(component)
    return instances


def calculate_binary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_flat = pred_mask.reshape(-1)
    gt_flat = gt_mask.reshape(-1)

    tp = np.sum((pred_flat == 1) & (gt_flat == 1))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return float(2 * (precision * recall) / (precision + recall))


def optimal_f1_score(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> float:
    # Authentic images only receive full credit when both sets are empty; any predicted
    # component on a pristine image should count as a complete failure for oF1.
    if not pred_masks and not gt_masks:
        return 1.0
    if not pred_masks or not gt_masks:
        return 0.0

    f1_matrix = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    for pred_idx, pred_mask in enumerate(pred_masks):
        for gt_idx, gt_mask in enumerate(gt_masks):
            f1_matrix[pred_idx, gt_idx] = calculate_binary_f1(pred_mask, gt_mask)

    if f1_matrix.shape[0] < len(gt_masks):
        pad_rows = len(gt_masks) - f1_matrix.shape[0]
        f1_matrix = np.vstack((f1_matrix, np.zeros((pad_rows, f1_matrix.shape[1]), dtype=np.float32)))

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-f1_matrix)
    # Extra predicted components are penalized separately so one good match cannot hide a
    # mask that fragments into many small blobs.
    excess_predictions_penalty = len(gt_masks) / max(len(pred_masks), len(gt_masks))
    return float(np.mean(f1_matrix[row_ind, col_ind]) * excess_predictions_penalty)
