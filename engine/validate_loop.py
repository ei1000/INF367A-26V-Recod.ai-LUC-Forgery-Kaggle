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
    validation_transfer_mode: str = "per_batch",
    post_process_confident_threshold: float | None = 0.9,
    post_process_smooth_probabilities: bool = True,
    post_process_fill_holes: bool = True,
    post_process_apply_opening: bool = True,
    post_process_apply_closing: bool = True,
    post_process_keep_confident_seeded_components: bool = False,
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
        transfer_mode=validation_transfer_mode,
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
        confident_threshold=post_process_confident_threshold,
        smooth_probabilities=post_process_smooth_probabilities,
        fill_holes=post_process_fill_holes,
        apply_opening=post_process_apply_opening,
        apply_closing=post_process_apply_closing,
        keep_confident_seeded_components=post_process_keep_confident_seeded_components,
    )
    result["inference_seconds"] = float(inference_seconds)
    result["validation_inference_mode"] = inference_mode
    result["probability_dtype"] = probability_dtype
    result["validation_transfer_mode"] = validation_transfer_mode

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
