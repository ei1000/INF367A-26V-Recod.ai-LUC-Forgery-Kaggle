from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import RandomSampler

from dataset_utils import SampleRecord
from engine.validation_records import (
    build_solution_rows,
    build_submission_rows,
    compute_kaggle_score_from_instances,
    compute_kaggle_score_via_recodai,
    connected_components_to_masks,
    load_resized_instance_masks,
)
from inference.postprocess import post_process_prediction
from recodai_f1 import calculate_f1_score


@dataclass(frozen=True, slots=True)
class ValidationPrediction:
    sample: SampleRecord
    probability: np.ndarray
    gt_union_mask: np.ndarray | None = None


def _probability_dtypes(probability_dtype: str) -> tuple[torch.dtype, np.dtype]:
    if probability_dtype == "float16":
        return torch.float16, np.dtype(np.float16)
    if probability_dtype == "float32":
        return torch.float32, np.dtype(np.float32)
    raise ValueError(f"probability_dtype must be 'float16' or 'float32', got {probability_dtype!r}")


def _squeeze_probability_batch(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities)
    if probs.ndim == 4 and probs.shape[1] == 1:
        return np.squeeze(probs, axis=1)
    if probs.ndim == 3:
        return probs
    raise ValueError(f"probability batch must have shape (B, 1, H, W) or (B, H, W), got {probs.shape}")


def _squeeze_probability_tensor(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim == 4 and probabilities.shape[1] == 1:
        return probabilities.squeeze(1)
    if probabilities.ndim == 3:
        return probabilities
    raise ValueError(
        f"probability batch must have shape (B, 1, H, W) or (B, H, W), got {tuple(probabilities.shape)}"
    )


@torch.no_grad()
def collect_validation_predictions(
    model: torch.nn.Module,
    val_loader,
    val_samples: Sequence[SampleRecord],
    device: torch.device,
    inference_mode: str,
    sliding_window_fn: Callable | None,
    probability_dtype: str = "float16",
    collect_masks: bool = False,
    transfer_mode: str = "per_batch",
) -> list[ValidationPrediction]:
    model.eval()

    torch_dtype, numpy_dtype = _probability_dtypes(probability_dtype)
    predictions: list[ValidationPrediction] = []
    sample_index = 0
    pending_samples: list[SampleRecord] = []
    pending_gt_union_masks: list[np.ndarray] | None = [] if collect_masks else None
    accumulated_probability_batch: torch.Tensor | None = None
    accumulated_probability_shape: tuple[int, ...] | None = None
    accumulated_probability_write_index = 0

    if isinstance(getattr(val_loader, "sampler", None), RandomSampler):
        raise ValueError(
            "val_loader must use shuffle=False: sample index mapping to val_samples requires sequential order"
        )
    if transfer_mode not in {"per_batch", "accumulate_gpu"}:
        raise ValueError(
            f"Unknown transfer_mode {transfer_mode!r}; expected 'per_batch' or 'accumulate_gpu'"
        )

    for imgs, masks in val_loader:
        batch_size = int(imgs.shape[0])
        batch_samples = val_samples[sample_index : sample_index + batch_size]
        if len(batch_samples) != batch_size:
            raise AssertionError(
                f"val_loader yielded more samples than val_samples: batch_size={batch_size}, "
                f"remaining={len(val_samples) - sample_index}"
            )

        if inference_mode == "direct":
            imgs_device = imgs.to(device, non_blocking=True)
            logits = model(imgs_device)
            probability_batch_tensor = _squeeze_probability_tensor(torch.sigmoid(logits).to(torch_dtype)).detach()
        elif inference_mode == "sliding":
            if sliding_window_fn is None:
                raise ValueError("sliding_window_fn is required when inference_mode='sliding'")
            imgs_device = imgs.to(device, non_blocking=True)
            batch_probabilities: list[torch.Tensor] = []
            for img in imgs_device:
                logits = sliding_window_fn(img, model, device)
                batch_probabilities.append(_squeeze_probability_tensor(torch.sigmoid(logits).to(torch_dtype)))
            probability_batch_tensor = _squeeze_probability_tensor(torch.stack(batch_probabilities, dim=0)).detach()
        else:
            raise ValueError(f"Unknown inference_mode {inference_mode!r}")

        gt_union_masks = None
        if collect_masks:
            gt_union_masks = masks.detach().cpu().numpy()[:, 0].astype(np.uint8, copy=False)

        if transfer_mode == "per_batch":
            probability_batch = probability_batch_tensor.cpu().numpy().astype(numpy_dtype, copy=False)
            for offset, sample in enumerate(batch_samples):
                gt_union_mask = gt_union_masks[offset] if gt_union_masks is not None else None
                predictions.append(
                    ValidationPrediction(
                        sample=sample,
                        probability=probability_batch[offset],
                        gt_union_mask=gt_union_mask,
                    )
                )
        elif transfer_mode == "accumulate_gpu":
            pending_samples.extend(batch_samples)
            if pending_gt_union_masks is not None and gt_union_masks is not None:
                pending_gt_union_masks.extend(gt_union_masks)

            batch_probability_shape = tuple(probability_batch_tensor.shape[1:])
            if accumulated_probability_batch is None:
                accumulated_probability_shape = batch_probability_shape
                accumulated_probability_batch = torch.empty(
                    (len(val_samples),) + batch_probability_shape,
                    dtype=probability_batch_tensor.dtype,
                    device=probability_batch_tensor.device,
                )
            elif batch_probability_shape != accumulated_probability_shape:
                raise ValueError(
                    "Validation probability shape changed across batches; "
                    f"expected {accumulated_probability_shape}, got {batch_probability_shape}"
                )
            assert accumulated_probability_batch is not None
            accumulated_probability_batch[
                accumulated_probability_write_index : accumulated_probability_write_index + batch_size
            ].copy_(probability_batch_tensor)
            accumulated_probability_write_index += batch_size

        sample_index += batch_size

    if sample_index != len(val_samples):
        raise AssertionError(
            f"val_loader yielded {sample_index} samples but val_samples has {len(val_samples)}"
        )

    if transfer_mode == "accumulate_gpu":
        if accumulated_probability_batch is not None:
            probability_batch = accumulated_probability_batch.cpu().numpy().astype(numpy_dtype, copy=False)
        else:
            probability_batch = np.empty((0,), dtype=numpy_dtype)

        for offset, sample in enumerate(pending_samples):
            gt_union_mask = pending_gt_union_masks[offset] if pending_gt_union_masks is not None else None
            predictions.append(
                ValidationPrediction(
                    sample=sample,
                    probability=probability_batch[offset],
                    gt_union_mask=gt_union_mask,
                )
            )

    return predictions


def _post_process_probability(
    probability: np.ndarray,
    pixel_util,
    pred_threshold: float,
    harden_temperature: float,
    hard_clip_low: float,
    hard_clip_high: float,
    min_component_area: int,
    confident_threshold: float | None = 0.9,
    smooth_probabilities: bool = True,
    fill_holes: bool = True,
    apply_opening: bool = True,
    apply_closing: bool = True,
    keep_confident_seeded_components: bool = False,
) -> np.ndarray:
    probs = np.asarray(probability, dtype=np.float32)
    if pixel_util is None:
        return (probs >= pred_threshold).astype(np.uint8)

    processed = post_process_prediction(
        probs=probs,
        pixel_util=pixel_util,
        threshold=pred_threshold,
        harden_temperature=harden_temperature,
        hard_clip_low=hard_clip_low,
        hard_clip_high=hard_clip_high,
        min_component_area=min_component_area,
        confident_threshold=confident_threshold,
        smooth_probabilities=smooth_probabilities,
        fill_holes=fill_holes,
        apply_opening=apply_opening,
        apply_closing=apply_closing,
        keep_confident_seeded_components=keep_confident_seeded_components,
    )
    return (processed >= pred_threshold).astype(np.uint8)


def score_validation_predictions(
    predictions: Sequence[ValidationPrediction],
    pixel_util,
    pred_threshold: float,
    harden_temperature: float,
    hard_clip_low: float,
    hard_clip_high: float,
    min_component_area: int,
    compute_pixel_f1: bool = False,
    verify_score_equivalence: bool = False,
    confident_threshold: float | None = 0.9,
    smooth_probabilities: bool = True,
    fill_holes: bool = True,
    apply_opening: bool = True,
    apply_closing: bool = True,
    keep_confident_seeded_components: bool = False,
) -> dict:
    postprocess_start = time.perf_counter()

    ordered_samples = [prediction.sample for prediction in predictions]
    pred_instances_by_sample_id: dict[str, list[np.ndarray]] = {}
    gt_instances_by_sample_id: dict[str, list[np.ndarray]] = {}
    shapes_by_sample_id: dict[str, tuple[int, int]] = {}
    pixel_f1s: list[float] = []

    for prediction in predictions:
        pred_bin = _post_process_probability(
            probability=prediction.probability,
            pixel_util=pixel_util,
            pred_threshold=pred_threshold,
            harden_temperature=harden_temperature,
            hard_clip_low=hard_clip_low,
            hard_clip_high=hard_clip_high,
            min_component_area=min_component_area,
            confident_threshold=confident_threshold,
            smooth_probabilities=smooth_probabilities,
            fill_holes=fill_holes,
            apply_opening=apply_opening,
            apply_closing=apply_closing,
            keep_confident_seeded_components=keep_confident_seeded_components,
        )
        sample = prediction.sample
        pred_instances_by_sample_id[sample.sample_id] = connected_components_to_masks(pred_bin)
        gt_instances_by_sample_id[sample.sample_id] = load_resized_instance_masks(sample, pred_bin.shape)
        shapes_by_sample_id[sample.sample_id] = pred_bin.shape

        if compute_pixel_f1:
            if prediction.gt_union_mask is None:
                raise ValueError("gt_union_mask is required when compute_pixel_f1=True")
            gt_bin = (prediction.gt_union_mask >= 0.5).astype(np.uint8)
            pixel_f1s.append(calculate_f1_score(pred_bin, gt_bin))

    postprocess_seconds = time.perf_counter() - postprocess_start
    scoring_start = time.perf_counter()

    kaggle_score = compute_kaggle_score_from_instances(
        ordered_samples,
        pred_instances_by_sample_id,
        gt_instances_by_sample_id,
    )

    if verify_score_equivalence:
        solution = build_solution_rows(ordered_samples, gt_instances_by_sample_id, shapes_by_sample_id)
        submission = build_submission_rows(ordered_samples, pred_instances_by_sample_id)
        official_score = compute_kaggle_score_via_recodai(solution, submission)
        if abs(float(kaggle_score) - float(official_score)) > 1e-8:
            raise AssertionError(
                f"Score equivalence check failed: direct={kaggle_score}, official={official_score}"
            )

    scoring_seconds = time.perf_counter() - scoring_start
    pixel_f1 = float(np.mean(pixel_f1s)) if compute_pixel_f1 else None

    return {
        "kaggle_score": float(kaggle_score),
        "pixel_f1": pixel_f1,
        "num_samples": len(ordered_samples),
        "num_forged": sum(1 for sample in ordered_samples if sample.label == "forged"),
        "num_authentic": sum(1 for sample in ordered_samples if sample.label == "authentic"),
        "postprocess_seconds": float(postprocess_seconds),
        "scoring_seconds": float(scoring_seconds),
    }
