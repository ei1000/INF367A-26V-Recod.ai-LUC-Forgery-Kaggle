from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from dataset import resolve_data_root, resolve_image_transform
from pipeline_helpers import (
    build_patchmatch_feature_branch,
    build_patchmatch_head,
    build_seunet_feature_branch,
    build_seunet_head,
    post_process_predictions,
)
from prediction.pixelmaputil_mask import MaskUtil
from training.checkpointing import load_module_state


def resolve_runtime_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_inference_transform(
    *,
    feature_backbone: str,
    use_dino_transform: bool,
    separate_transforms: bool,
):
    return resolve_image_transform(
        feature_backbone=feature_backbone,
        use_dino_transform=use_dino_transform,
        cnn_backbone="pretrained",
        separate_transforms=separate_transforms,
    )


def ensure_checkpoint_module_loaded(module, state_dict: dict[str, torch.Tensor], module_name: str) -> None:
    if not load_module_state(module, state_dict, module_name):
        raise RuntimeError(f"Failed to fully restore {module_name} from checkpoint.")


def restore_feature_branches_from_checkpoint(
    checkpoint: dict[str, object],
    device: torch.device,
    *,
    feature_backbone: str,
    dino_model_name: str,
    separate_transforms: bool,
    use_dino_transform: bool,
):
    pm_backbone, pyramid_zm = build_patchmatch_feature_branch(device)
    dino_extractor = build_seunet_feature_branch(
        device,
        feature_backbone=feature_backbone,
        dino_model_name=dino_model_name,
        separate_transforms=separate_transforms,
        use_dino_transform=use_dino_transform,
    )

    if "pm_backbone" not in checkpoint:
        raise KeyError("Checkpoint is missing pm_backbone weights.")
    ensure_checkpoint_module_loaded(pm_backbone, checkpoint["pm_backbone"], "pm_backbone")

    if "dino_extractor" in checkpoint:
        ensure_checkpoint_module_loaded(dino_extractor, checkpoint["dino_extractor"], "dino_extractor")
    elif "pyramid_bb" in checkpoint:
        ensure_checkpoint_module_loaded(dino_extractor, checkpoint["pyramid_bb"], "dino_extractor")
    else:
        raise KeyError("Checkpoint is missing dino_extractor/pyramid_bb weights.")

    return pm_backbone, dino_extractor, pyramid_zm


def restore_localization_heads_from_checkpoint(
    checkpoint: dict[str, object],
    *,
    cnn_errors: torch.Tensor,
    dino_features: torch.Tensor,
    device: torch.device,
):
    dlf_decoder = build_patchmatch_head(cnn_errors, device)
    ensure_checkpoint_module_loaded(dlf_decoder, checkpoint["dlf_decoder"], "dlf_decoder")
    dlf_decoder.eval()

    se_model = build_seunet_head(dino_features, device)
    ensure_checkpoint_module_loaded(se_model, checkpoint["se_model"], "se_model")
    se_model.eval()
    return dlf_decoder, se_model


def predict_binary_mask(
    refined_mask: torch.Tensor,
    *,
    disable_post_process: bool,
    raw_threshold: float,
    post_process_threshold: float,
    post_process_confident_threshold: float | None,
    post_process_min_component_area: int,
    post_process_smooth_probabilities: bool,
    post_process_fill_holes: bool,
    post_process_apply_closing: bool,
    util: MaskUtil | None = None,
) -> torch.Tensor:
    if disable_post_process:
        return (refined_mask.squeeze(1) >= raw_threshold).long()

    if util is None:
        util = MaskUtil()

    return post_process_predictions(
        refined_mask,
        util,
        do_post_process=True,
        post_process_threshold=post_process_threshold,
        post_process_confident_threshold=post_process_confident_threshold,
        post_process_min_component_area=post_process_min_component_area,
        post_process_smooth_probabilities=post_process_smooth_probabilities,
        post_process_fill_holes=post_process_fill_holes,
        post_process_apply_closing=post_process_apply_closing,
    )


def safe_prediction_stem(sample_path: str | Path) -> str:
    path = Path(sample_path)
    data_root = resolve_data_root().resolve()
    try:
        relative = path.resolve().relative_to(data_root)
        return "__".join(relative.with_suffix("").parts)
    except ValueError:
        return path.stem


def load_display_image(path: Path, size: int | tuple[int, int]) -> np.ndarray:
    if isinstance(size, int):
        size = (size, size)
    image = Image.open(path).convert("RGB").resize((size[1], size[0]), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0
