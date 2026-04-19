from __future__ import annotations

import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from dataset_utils import SampleRecord, load_instance_masks
from recodai_f1 import oF1_score, rle_encode, score


def resize_binary_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    pil_mask = Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L")
    resized = pil_mask.resize((width, height), resample=Image.Resampling.NEAREST)
    return (np.asarray(resized) > 0).astype(np.uint8)


def load_resized_instance_masks(sample: SampleRecord, shape: tuple[int, int]) -> list[np.ndarray]:
    if sample.label == "authentic":
        return []
    if len(sample.mask_paths) == 0:
        raise ValueError(
            f"forged sample {sample.sample_id} case_id={sample.case_id} is missing instance mask paths"
        )
    return [resize_binary_mask(mask, shape) for mask in load_instance_masks(sample.mask_paths)]


def connected_components_to_masks(mask: np.ndarray) -> list[np.ndarray]:
    labeled, num_components = ndimage.label(np.asarray(mask).astype(np.uint8) > 0)
    return [(labeled == component_id).astype(np.uint8) for component_id in range(1, num_components + 1)]


def mask_instances_to_annotation(instances) -> str:
    instances_list = list(instances)
    if len(instances_list) == 0:
        return "authentic"
    return rle_encode(instances_list)


def _normalize_prediction_instances(
    pred_instances_by_sample_id: Mapping[str, Sequence[np.ndarray]],
    sample_id: str,
) -> list[np.ndarray]:
    instances = pred_instances_by_sample_id.get(sample_id, [])
    return list(instances) if len(instances) > 0 else []


def _get_ground_truth_instances(
    sample: SampleRecord,
    gt_instances_by_sample_id: Mapping[str, Sequence[np.ndarray]],
) -> list[np.ndarray]:
    instances = gt_instances_by_sample_id.get(sample.sample_id)
    if sample.label == "authentic":
        if instances is None or len(instances) == 0:
            return []
        raise ValueError(f"authentic sample {sample.sample_id} must have empty ground truth instances")

    if instances is None:
        raise KeyError(f"missing ground truth instances for forged sample {sample.sample_id}")
    if len(instances) == 0:
        raise ValueError(f"forged sample {sample.sample_id} must have non-empty ground truth instances")
    return list(instances)


def score_instances(pred_instances: Sequence[np.ndarray], gt_instances: Sequence[np.ndarray]) -> float:
    pred_empty = len(pred_instances) == 0
    gt_empty = len(gt_instances) == 0
    if gt_empty and pred_empty:
        return 1.0
    if gt_empty and not pred_empty:
        return 0.0
    if not gt_empty and pred_empty:
        return 0.0
    return float(oF1_score(list(pred_instances), list(gt_instances)))


def compute_kaggle_score_from_instances(
    ordered_samples: Sequence[SampleRecord],
    pred_instances_by_sample_id: Mapping[str, Sequence[np.ndarray]],
    gt_instances_by_sample_id: Mapping[str, Sequence[np.ndarray]],
) -> float:
    scores = [
        score_instances(
            pred_instances=_normalize_prediction_instances(pred_instances_by_sample_id, sample.sample_id),
            gt_instances=_get_ground_truth_instances(sample, gt_instances_by_sample_id),
        )
        for sample in ordered_samples
    ]
    return float(np.mean(scores)) if scores else 0.0


def build_solution_rows(
    ordered_samples: Sequence[SampleRecord],
    gt_instances_by_sample_id: Mapping[str, Sequence[np.ndarray]],
    shapes_by_sample_id: Mapping[str, tuple[int, int]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample in ordered_samples:
        if sample.label == "authentic":
            gt_instances = gt_instances_by_sample_id.get(sample.sample_id)
            if gt_instances is not None and len(gt_instances) > 0:
                raise ValueError(f"authentic sample {sample.sample_id} must have empty ground truth instances")
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "annotation": "authentic",
                    "shape": "authentic",
                }
            )
            continue

        shape = shapes_by_sample_id[sample.sample_id]
        gt_instances = _get_ground_truth_instances(sample, gt_instances_by_sample_id)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "annotation": mask_instances_to_annotation(gt_instances),
                "shape": json.dumps([shape[0], shape[1]]),
            }
        )
    return pd.DataFrame(rows, columns=["sample_id", "annotation", "shape"])


def build_submission_rows(
    ordered_samples: Sequence[SampleRecord],
    pred_instances_by_sample_id: Mapping[str, Sequence[np.ndarray]],
) -> pd.DataFrame:
    rows = [
        {
            "sample_id": sample.sample_id,
            "annotation": mask_instances_to_annotation(
                _normalize_prediction_instances(pred_instances_by_sample_id, sample.sample_id)
            ),
        }
        for sample in ordered_samples
    ]
    return pd.DataFrame(rows, columns=["sample_id", "annotation"])


def compute_kaggle_score_via_recodai(solution: pd.DataFrame, submission: pd.DataFrame) -> float:
    return float(score(solution.copy(), submission.copy(), row_id_column_name="sample_id"))
