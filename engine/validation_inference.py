from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
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


def _numpy_dtype(probability_dtype: str) -> np.dtype:
    if probability_dtype == "float16":
        return np.dtype(np.float16)
    if probability_dtype == "float32":
        return np.dtype(np.float32)
    raise ValueError(f"probability_dtype must be 'float16' or 'float32', got {probability_dtype!r}")


def _squeeze_probability_batch(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities)
    if probs.ndim == 4 and probs.shape[1] == 1:
        return np.squeeze(probs, axis=1)
    if probs.ndim == 3:
        return probs
    raise ValueError(f"probability batch must have shape (B, 1, H, W) or (B, H, W), got {probs.shape}")


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
) -> list[ValidationPrediction]:
    model.eval()

    numpy_dtype = _numpy_dtype(probability_dtype)
    predictions: list[ValidationPrediction] = []
    sample_index = 0

    if isinstance(getattr(val_loader, "sampler", None), RandomSampler):
        raise ValueError(
            "val_loader must use shuffle=False: sample index mapping to val_samples requires sequential order"
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
            probs = torch.sigmoid(logits)
            probability_batch = _squeeze_probability_batch(probs.detach().cpu().numpy()).astype(
                numpy_dtype, copy=False
            )
        elif inference_mode == "sliding":
            if sliding_window_fn is None:
                raise ValueError("sliding_window_fn is required when inference_mode='sliding'")
            imgs_device = imgs.to(device, non_blocking=True)
            batch_probabilities: list[torch.Tensor] = []
            for img in imgs_device:
                logits = sliding_window_fn(img, model, device)
                probs = torch.sigmoid(logits)
                batch_probabilities.append(torch.squeeze(probs, dim=0) if probs.ndim == 4 else probs)
            probability_batch = _squeeze_probability_batch(torch.stack(batch_probabilities, dim=0).detach().cpu().numpy()).astype(
                numpy_dtype,
                copy=False,
            )
        else:
            raise ValueError(f"Unknown inference_mode {inference_mode!r}")

        gt_union_masks = None
        if collect_masks:
            gt_union_masks = masks.detach().cpu().numpy()[:, 0].astype(np.uint8, copy=False)

        for offset, sample in enumerate(batch_samples):
            gt_union_mask = gt_union_masks[offset] if gt_union_masks is not None else None
            predictions.append(
                ValidationPrediction(
                    sample=sample,
                    probability=probability_batch[offset],
                    gt_union_mask=gt_union_mask,
                )
            )

        sample_index += batch_size

    if sample_index != len(val_samples):
        raise AssertionError(
            f"val_loader yielded {sample_index} samples but val_samples has {len(val_samples)}"
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
    )
    return (processed >= pred_threshold).astype(np.uint8)


def _save_prediction_batches(
    predictions: Sequence[ValidationPrediction],
    prediction_masks: Sequence[np.ndarray],
    output_dir: Path,
    filename_prefix: str,
    batch_size: int,
) -> tuple[int, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / f"{filename_prefix}_index.jsonl"
    index_lines: list[str] = []

    for batch_start in range(0, len(predictions), batch_size):
        batch_predictions = predictions[batch_start : batch_start + batch_size]
        batch_masks = prediction_masks[batch_start : batch_start + batch_size]
        batch_idx = (batch_start // batch_size) + 1
        batch_tensor = torch.from_numpy(np.stack(batch_masks, axis=0).astype(np.uint8, copy=False))
        batch_path = output_dir / f"{filename_prefix}_batch_{batch_idx:05d}.pt"
        torch.save(batch_tensor, batch_path)

        for mask_idx, prediction in enumerate(batch_predictions):
            index_lines.append(
                json.dumps(
                    {
                        "prediction_file": str(batch_path),
                        "mask_index": mask_idx,
                        "sample_id": prediction.sample.sample_id,
                        "image_path": str(prediction.sample.image_path),
                        "label": prediction.sample.label,
                    }
                )
            )

    index_path.write_text("\n".join(index_lines), encoding="utf-8")
    saved_batches = (len(predictions) + max(batch_size, 1) - 1) // max(batch_size, 1)
    return saved_batches, str(index_path)


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
    prediction_output_dir: str | Path | None = None,
    prediction_filename_prefix: str | None = None,
    prediction_batch_size: int | None = None,
) -> dict:
    postprocess_start = time.perf_counter()

    ordered_samples = [prediction.sample for prediction in predictions]
    pred_instances_by_sample_id: dict[str, list[np.ndarray]] = {}
    gt_instances_by_sample_id: dict[str, list[np.ndarray]] = {}
    shapes_by_sample_id: dict[str, tuple[int, int]] = {}
    pixel_f1s: list[float] = []
    saved_prediction_masks: list[np.ndarray] | None = [] if prediction_output_dir is not None else None

    for prediction in predictions:
        pred_bin = _post_process_probability(
            probability=prediction.probability,
            pixel_util=pixel_util,
            pred_threshold=pred_threshold,
            harden_temperature=harden_temperature,
            hard_clip_low=hard_clip_low,
            hard_clip_high=hard_clip_high,
            min_component_area=min_component_area,
        )
        sample = prediction.sample
        pred_instances_by_sample_id[sample.sample_id] = connected_components_to_masks(pred_bin)
        gt_instances_by_sample_id[sample.sample_id] = load_resized_instance_masks(sample, pred_bin.shape)
        shapes_by_sample_id[sample.sample_id] = pred_bin.shape
        if saved_prediction_masks is not None:
            saved_prediction_masks.append(pred_bin)

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
    saved_prediction_batches = 0
    prediction_index_path = None
    if saved_prediction_masks is not None:
        saved_prediction_batches, prediction_index_path = _save_prediction_batches(
            predictions=predictions,
            prediction_masks=saved_prediction_masks,
            output_dir=Path(prediction_output_dir),
            filename_prefix=prediction_filename_prefix or "predictions",
            batch_size=max(int(prediction_batch_size or 1), 1),
        )

    return {
        "kaggle_score": float(kaggle_score),
        "pixel_f1": pixel_f1,
        "num_samples": len(ordered_samples),
        "num_forged": sum(1 for sample in ordered_samples if sample.label == "forged"),
        "num_authentic": sum(1 for sample in ordered_samples if sample.label == "authentic"),
        "postprocess_seconds": float(postprocess_seconds),
        "scoring_seconds": float(scoring_seconds),
        "saved_prediction_batches": int(saved_prediction_batches),
        "prediction_index_path": prediction_index_path,
    }
