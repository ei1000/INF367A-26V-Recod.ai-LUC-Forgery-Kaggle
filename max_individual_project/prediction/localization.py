from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from cross_scale_patchmatch.pixel_propagator import PixelPropagator
from dataset import imagenet_normalize_tensor
from datatypes import DLFDecoderInput, PatchMatchBranchResult
from prediction.multi_scale_dlf import MultiScaleDLF


def synchronize_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def normalize_dlf_error_maps(
    error_maps: torch.Tensor,
    mode: str = "log1p",
    eps: float = 1e-6,
) -> torch.Tensor:
    if error_maps.dim() != 4:
        raise ValueError(f"Expected error maps shape [B,S,H,W], got {tuple(error_maps.shape)}")

    error_maps = torch.clamp_min(error_maps.float(), 0.0)
    if mode == "none":
        return error_maps
    if mode == "log1p":
        return torch.log1p(error_maps)
    if mode == "zscore":
        error_maps = torch.log1p(error_maps)
        flattened = error_maps.flatten(start_dim=2)
        mean = flattened.mean(dim=-1, keepdim=True).unsqueeze(-1)
        std = flattened.std(dim=-1, keepdim=True, unbiased=False).unsqueeze(-1)
        normalized = (error_maps - mean) / (std + eps)
        return normalized.clamp(-6.0, 6.0)

    raise ValueError(f"Unsupported DLF error scaling mode: {mode}")


def select_primary_feature_map(feature_maps):
    if isinstance(feature_maps, (tuple, list)):
        if not feature_maps:
            raise ValueError("Expected at least one feature map.")
        return feature_maps[len(feature_maps) // 2]
    return feature_maps


def extract_localization_inputs(
    images: torch.Tensor,
    pm_backbone,
    pyramid_zm,
    dino_extractor,
    separate_transforms: bool,
    cnn_feature_norm: bool,
    pm_random_window: int,
    pm_iters: int,
    pm_beta: float,
    pm_hard_selection: bool,
    pm_use_non_local: bool,
    pm_non_local_limit: float,
    pm_reduced_precision: bool = True,
    localization_resolution: str = "image",
    dlf_error_scaling: str = "log1p",
    collect_stats: bool = False,
    pm_flat_threshold: float = 0.15,
    pm_margin_threshold: float = 0.10,
):
    """Build the three-branch localization inputs.

    Branch layout:
    - frozen ResNet18 descriptors for PatchMatch
    - multi-scale Zernike descriptors for PatchMatch
    - frozen DINO features for the SEUNet refinement branch
    """

    if localization_resolution != "image":
        raise ValueError(
            "This pipeline now computes PatchMatch and DLF on the image grid only. "
            f"Got localization_resolution={localization_resolution!r}."
        )

    images_backbone = images
    if separate_transforms:
        images_backbone = imagenet_normalize_tensor(images)

    device = images.device
    localization_stats = None
    peak_memory_base = None
    if collect_stats and device.type == "cuda":
        synchronize_if_cuda(device)
        torch.cuda.reset_peak_memory_stats(device)
        peak_memory_base = torch.cuda.memory_allocated(device)

    if collect_stats:
        synchronize_if_cuda(device)
        feature_start = time.perf_counter()

    with torch.no_grad():
        cnn_feats = pm_backbone(images_backbone)
        if cnn_feature_norm:
            cnn_feats = tuple(F.normalize(feature, p=2, dim=1) for feature in cnn_feats)
    with torch.no_grad():
        zernike_feats = tuple(feature.detach() for feature in pyramid_zm(images))
    with torch.no_grad():
        dino_feature_maps = dino_extractor(images)
        dino_features = select_primary_feature_map(dino_feature_maps).detach()

    if collect_stats:
        synchronize_if_cuda(device)
        feature_time = time.perf_counter() - feature_start

    if collect_stats:
        synchronize_if_cuda(device)
        propagation_start = time.perf_counter()
    propagator = PixelPropagator(
        images,
        cnn_feats,
        zernike_feats,
        random_window=pm_random_window,
        reduced_precision=pm_reduced_precision,
    )
    del cnn_feats
    del zernike_feats
    batch_cnn_branch, batch_zernike_branch = propagator.propagation_layer(
        iters=pm_iters,
        beta=pm_beta,
        hard_selection=pm_hard_selection,
        use_non_local=pm_use_non_local,
        non_local_limit=pm_non_local_limit,
        flat_threshold=pm_flat_threshold,
        margin_threshold=pm_margin_threshold,
    )

    if collect_stats:
        synchronize_if_cuda(device)
        propagation_time = time.perf_counter() - propagation_start

    if collect_stats:
        synchronize_if_cuda(device)
        dlf_start = time.perf_counter()

    cnn_errors = MultiScaleDLF(
        images,
        batch_cnn_branch.offsets,
    ).compute_errors()
    zernike_errors = MultiScaleDLF(
        images,
        batch_zernike_branch.offsets,
    ).compute_errors()
    cnn_errors = normalize_dlf_error_maps(cnn_errors, mode=dlf_error_scaling)
    zernike_errors = normalize_dlf_error_maps(zernike_errors, mode=dlf_error_scaling)
    if collect_stats:
        synchronize_if_cuda(device)
        dlf_time = time.perf_counter() - dlf_start
        localization_stats = {
            "feature_time_s": feature_time,
            "patchmatch_time_s": propagation_time,
            "dlf_time_s": dlf_time,
        }
        if peak_memory_base is not None:
            peak_bytes = torch.cuda.max_memory_allocated(device) - peak_memory_base
            localization_stats["localization_peak_memory_mb"] = peak_bytes / (1024 ** 2)

    return cnn_errors, zernike_errors, batch_cnn_branch, batch_zernike_branch, dino_features, localization_stats


def decode_and_refine_masks(
    images: torch.Tensor,
    cnn_error_maps: torch.Tensor,
    zernike_error_maps: torch.Tensor,
    cnn_branch_result: PatchMatchBranchResult,
    zernike_branch_result: PatchMatchBranchResult,
    dlf_decoder,
    se_model,
    dino_features: torch.Tensor,
    output_size: tuple[int, int] | None = None,
):
    dlf_decoder_input = DLFDecoderInput(
        cnn_error_maps=cnn_error_maps,
        zernike_error_maps=zernike_error_maps,
        cnn_offsets=cnn_branch_result.offsets,
        zernike_offsets=zernike_branch_result.offsets,
    )

    dlf_map = dlf_decoder(dlf_decoder_input)
    target_map = se_model(dino_features)
    if target_map.shape[-2:] != dlf_map.shape[-2:]:
        target_map = F.interpolate(target_map, size=dlf_map.shape[-2:], mode="bilinear", align_corners=False)
    refined_mask = torch.maximum(dlf_map, target_map)

    if output_size is None:
        output_size = images.shape[-2:]
    if refined_mask.shape[-2:] != output_size:
        refined_mask = F.interpolate(refined_mask, size=output_size, mode="bilinear", align_corners=False)
        target_map = F.interpolate(target_map, size=output_size, mode="bilinear", align_corners=False)
        dlf_map = F.interpolate(dlf_map, size=output_size, mode="bilinear", align_corners=False)
    return refined_mask, target_map, dlf_map


def run_localization(
    images: torch.Tensor,
    pm_backbone,
    pyramid_zm,
    dino_extractor,
    dlf_decoder,
    se_model,
    *,
    separate_transforms: bool,
    cnn_feature_norm: bool,
    pm_random_window: int,
    pm_iters: int,
    pm_beta: float,
    pm_hard_selection: bool,
    pm_use_non_local: bool,
    pm_non_local_limit: float,
    pm_reduced_precision: bool = True,
    localization_resolution: str = "image",
    dlf_error_scaling: str = "log1p",
    collect_stats: bool = False,
    pm_flat_threshold: float = 0.15,
    pm_margin_threshold: float = 0.10,
):
    cnn_errors, zernike_errors, cnn_branch_result, zernike_branch_result, dino_features, localization_stats = (
        extract_localization_inputs(
            images=images,
            pm_backbone=pm_backbone,
            pyramid_zm=pyramid_zm,
            dino_extractor=dino_extractor,
            separate_transforms=separate_transforms,
            cnn_feature_norm=cnn_feature_norm,
            pm_random_window=pm_random_window,
            pm_iters=pm_iters,
            pm_beta=pm_beta,
            pm_hard_selection=pm_hard_selection,
            pm_use_non_local=pm_use_non_local,
            pm_non_local_limit=pm_non_local_limit,
            pm_reduced_precision=pm_reduced_precision,
            localization_resolution=localization_resolution,
            dlf_error_scaling=dlf_error_scaling,
            collect_stats=collect_stats,
            pm_flat_threshold=pm_flat_threshold,
            pm_margin_threshold=pm_margin_threshold,
        )
    )
    refined_mask, target_map, dlf_map = decode_and_refine_masks(
        images=images,
        cnn_error_maps=cnn_errors,
        zernike_error_maps=zernike_errors,
        cnn_branch_result=cnn_branch_result,
        zernike_branch_result=zernike_branch_result,
        dlf_decoder=dlf_decoder,
        se_model=se_model,
        dino_features=dino_features,
        output_size=images.shape[-2:],
    )
    return refined_mask, target_map, dlf_map, cnn_branch_result, zernike_branch_result, dino_features, cnn_errors, localization_stats
