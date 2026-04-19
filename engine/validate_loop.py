from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import RandomSampler
from tqdm import tqdm

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
) -> dict:
    model.eval()

    if isinstance(val_loader.sampler, RandomSampler):
        raise ValueError(
            "val_loader must use shuffle=False: sample index mapping to val_samples requires sequential order"
        )

    pred_instances_by_sample_id: dict[str, list[np.ndarray]] = {}
    gt_instances_by_sample_id: dict[str, list[np.ndarray]] = {}
    shapes_by_sample_id: dict[str, tuple[int, int]] = {}
    pixel_f1s: list[float] = []

    sample_index = 0

    for imgs, masks in tqdm(val_loader, desc=f"epoch {epoch_idx + 1} val"):
        imgs = imgs.to(device)
        masks = masks.to(device)

        for i in range(imgs.shape[0]):
            sample = val_samples[sample_index]
            sample_index += 1

            img = imgs[i]
            mask = masks[i]

            logits = sliding_window_fn(img, model, device)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs = post_process_prediction(
                probs=probs,
                pixel_util=pixel_util,
                threshold=pred_threshold,
                harden_temperature=harden_temperature,
                hard_clip_low=hard_clip_low,
                hard_clip_high=hard_clip_high,
                min_component_area=min_component_area,
            )

            pred_bin = (probs >= pred_threshold).astype(np.uint8)

            pred_instances_by_sample_id[sample.sample_id] = connected_components_to_masks(pred_bin)
            gt_instances_by_sample_id[sample.sample_id] = load_resized_instance_masks(sample, pred_bin.shape)
            shapes_by_sample_id[sample.sample_id] = pred_bin.shape

            if compute_pixel_f1:
                gt = mask.cpu().numpy()[0]
                gt_bin = (gt >= 0.5).astype(np.uint8)
                pixel_f1s.append(calculate_f1_score(pred_bin, gt_bin))

    assert sample_index == len(val_samples), (
        f"val_loader yielded {sample_index} samples but val_samples has {len(val_samples)}"
    )

    kaggle_score = compute_kaggle_score_from_instances(
        val_samples,
        pred_instances_by_sample_id,
        gt_instances_by_sample_id,
    )

    if verify_score_equivalence:
        solution = build_solution_rows(val_samples, gt_instances_by_sample_id, shapes_by_sample_id)
        submission = build_submission_rows(val_samples, pred_instances_by_sample_id)
        official_score = compute_kaggle_score_via_recodai(solution, submission)
        assert abs(float(kaggle_score) - float(official_score)) <= 1e-8, (
            f"Score equivalence check failed: direct={kaggle_score}, official={official_score}"
        )

    pixel_f1: float | None = float(np.mean(pixel_f1s)) if compute_pixel_f1 else None
    if compute_pixel_f1:
        print(
            f"[non-official] pixel F1 (epoch {epoch_idx + 1}): {pixel_f1:.4f}"
        )

    return {
        "kaggle_score": float(kaggle_score),
        "pixel_f1": pixel_f1,
        "num_samples": len(val_samples),
        "num_forged": sum(1 for s in val_samples if s.label == "forged"),
        "num_authentic": sum(1 for s in val_samples if s.label == "authentic"),
    }
