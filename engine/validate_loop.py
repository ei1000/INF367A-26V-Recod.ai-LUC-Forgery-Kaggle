from __future__ import annotations

import time
from typing import Sequence

import torch
from torch.utils.data import RandomSampler
from tqdm import tqdm

from dataset_utils import SampleRecord
from engine.validation_inference import collect_validation_predictions, score_validation_predictions


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    val_loader,
    val_samples: Sequence[SampleRecord],
    device: torch.device,
    sliding_window_fn,
    pixel_util,
    pred_threshold: float,
    harden_temperature: float,
    hard_clip_low: float,
    hard_clip_high: float,
    min_component_area: int,
    epoch_idx: int,
    compute_pixel_f1: bool = False,
    verify_score_equivalence: bool = False,
    inference_mode: str = "direct",
    probability_dtype: str = "float16",
    log_timing: bool = True,
) -> dict:
    if isinstance(getattr(val_loader, "sampler", None), RandomSampler):
        raise ValueError(
            "val_loader must use shuffle=False: sample index mapping to val_samples requires sequential order"
        )

    inference_start = time.perf_counter()
    progress_loader = tqdm(val_loader, desc=f"epoch {epoch_idx + 1} val inference")
    predictions = collect_validation_predictions(
        model=model,
        val_loader=progress_loader,
        val_samples=val_samples,
        device=device,
        inference_mode=inference_mode,
        sliding_window_fn=sliding_window_fn,
        probability_dtype=probability_dtype,
        collect_masks=compute_pixel_f1,
    )
    inference_seconds = time.perf_counter() - inference_start

    result = score_validation_predictions(
        predictions=predictions,
        pixel_util=pixel_util,
        pred_threshold=pred_threshold,
        harden_temperature=harden_temperature,
        hard_clip_low=hard_clip_low,
        hard_clip_high=hard_clip_high,
        min_component_area=min_component_area,
        compute_pixel_f1=compute_pixel_f1,
        verify_score_equivalence=verify_score_equivalence,
    )
    result["inference_seconds"] = float(inference_seconds)
    result["validation_inference_mode"] = inference_mode
    result["probability_dtype"] = probability_dtype

    if compute_pixel_f1:
        print(f"[non-official] pixel F1 (epoch {epoch_idx + 1}): {result['pixel_f1']:.4f}")

    if log_timing:
        print(
            "Validation timing "
            f"(epoch {epoch_idx + 1}): "
            f"inference={result['inference_seconds']:.2f}s "
            f"postprocess={result['postprocess_seconds']:.2f}s "
            f"scoring={result['scoring_seconds']:.2f}s"
        )

    return result
