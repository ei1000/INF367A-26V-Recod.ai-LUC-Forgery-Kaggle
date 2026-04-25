from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage


@dataclass(frozen=True, slots=True)
class DerivedSourceTargetMasks:
    union_mask: np.ndarray
    source_mask: np.ndarray
    target_mask: np.ndarray
    diff: np.ndarray | None
    diff_mask: np.ndarray | None
    component_scores: pd.DataFrame
    status: str


def derive_source_target_masks_from_arrays(
    authentic: np.ndarray | None,
    forged: np.ndarray,
    union_mask: np.ndarray,
    *,
    diff_threshold: float = 5.0,
    component_change_fraction: float = 0.25,
    min_component_area: int = 0,
) -> DerivedSourceTargetMasks:
    union_bool = np.asarray(union_mask) > 0

    if authentic is None:
        target_mask = union_bool.astype(np.uint8)
        source_mask = np.zeros_like(target_mask, dtype=np.uint8)
        return DerivedSourceTargetMasks(
            union_mask=target_mask,
            source_mask=source_mask,
            target_mask=target_mask,
            diff=None,
            diff_mask=None,
            component_scores=pd.DataFrame(),
            status="target_only_no_authentic",
        )

    diff = _abs_difference(authentic, forged)
    diff_mask = diff > diff_threshold
    source_mask, target_mask, component_scores = _split_union_components_by_diff(
        union_bool,
        diff,
        diff_mask,
        component_change_fraction=component_change_fraction,
        min_component_area=min_component_area,
    )

    return DerivedSourceTargetMasks(
        union_mask=union_bool.astype(np.uint8),
        source_mask=source_mask.astype(np.uint8),
        target_mask=target_mask.astype(np.uint8),
        diff=diff,
        diff_mask=diff_mask.astype(np.uint8),
        component_scores=component_scores,
        status="derived_from_pair",
    )


def _split_union_components_by_diff(
    union_mask: np.ndarray,
    diff: np.ndarray,
    diff_mask: np.ndarray,
    *,
    component_change_fraction: float,
    min_component_area: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    labels, label_count = ndimage.label(union_mask, structure=np.ones((3, 3), dtype=np.uint8))
    target_mask = np.zeros_like(union_mask, dtype=bool)
    source_mask = np.zeros_like(union_mask, dtype=bool)
    rows = []

    for label_idx in range(1, label_count + 1):
        component = labels == label_idx
        area = int(component.sum())
        changed_pixels = int(diff_mask[component].sum())
        changed_fraction = changed_pixels / area if area else 0.0
        mean_diff = float(diff[component].mean()) if area else 0.0
        max_diff = float(diff[component].max()) if area else 0.0
        is_tiny = area < min_component_area
        is_target = (not is_tiny) and changed_fraction >= component_change_fraction

        rows.append(
            {
                "component": label_idx,
                "area": area,
                "changed_pixels": changed_pixels,
                "changed_fraction": changed_fraction,
                "mean_diff": mean_diff,
                "max_diff": max_diff,
                "is_tiny": is_tiny,
                "is_target": is_target,
            }
        )

        if is_tiny:
            continue
        if is_target:
            target_mask |= component
        else:
            source_mask |= component

    component_scores = pd.DataFrame(rows)
    if target_mask.any() or component_scores.empty:
        return source_mask, target_mask, component_scores

    candidates = component_scores[~component_scores["is_tiny"]]
    if candidates.empty:
        return source_mask, target_mask, component_scores

    fallback_component = int(
        candidates.sort_values(["changed_fraction", "mean_diff", "max_diff", "area"], ascending=False).iloc[0]["component"]
    )
    fallback_mask = labels == fallback_component
    target_mask |= fallback_mask
    source_mask &= ~fallback_mask
    component_scores.loc[component_scores["component"] == fallback_component, "is_target"] = True
    return source_mask, target_mask, component_scores


def _abs_difference(authentic: np.ndarray, forged: np.ndarray) -> np.ndarray:
    authentic_gray = _as_float_gray(authentic)
    forged_gray = _as_float_gray(forged)
    if authentic_gray.shape != forged_gray.shape:
        raise ValueError(f"Authentic and forged images have different shapes: {authentic_gray.shape} vs {forged_gray.shape}")
    return np.abs(forged_gray - authentic_gray)


def _as_float_gray(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)
    return image
