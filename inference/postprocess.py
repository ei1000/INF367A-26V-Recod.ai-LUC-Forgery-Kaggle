import numpy as np
from scipy import ndimage

from util.pixelmapUtil import PixelMapUtil


def harden_probabilities(
    probs: np.ndarray,
    temperature: float,
    clip_low: float,
    clip_high: float,
) -> np.ndarray:
    eps = 1e-6
    probs = np.clip(probs, eps, 1.0 - eps)
    logits = np.log(probs / (1.0 - probs))
    hardened = 1.0 / (1.0 + np.exp(-(logits / temperature)))
    hardened = np.where(hardened <= clip_low, 0.0, hardened)
    hardened = np.where(hardened >= clip_high, 1.0, hardened)
    return hardened.astype(np.float32, copy=False)


def post_process_prediction(
    probs: np.ndarray,
    pixel_util: PixelMapUtil,
    threshold: float,
    harden_temperature: float,
    hard_clip_low: float,
    hard_clip_high: float,
    min_component_area: int = 0,
) -> np.ndarray:
    probs = harden_probabilities(
        probs,
        temperature=harden_temperature,
        clip_low=hard_clip_low,
        clip_high=hard_clip_high,
    )
    mask = pixel_util.post_process_mask_probs(probs, threshold=threshold)
    return filter_small_components(mask, min_component_area=min_component_area)


def filter_small_components(mask: np.ndarray, min_component_area: int) -> np.ndarray:
    if min_component_area <= 1:
        return mask.astype(np.float32, copy=False)

    mask_bool = mask.astype(bool)
    labeled, num = ndimage.label(mask_bool)
    if num == 0:
        return mask.astype(np.float32, copy=False)

    component_ids = np.arange(1, num + 1)
    areas = ndimage.sum(mask_bool, labeled, index=component_ids)

    keep = np.zeros_like(mask_bool, dtype=bool)
    for comp_id, area in zip(component_ids, areas):
        if area >= min_component_area:
            keep |= labeled == comp_id

    return keep.astype(np.float32)
