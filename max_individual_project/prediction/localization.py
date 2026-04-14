from __future__ import annotations

import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from cross_scale_patternmatch.pixel_propagator import PixelPropagator
from dataset import imagenet_normalize_tensor
from datatypes import DLFDecoderInput, PatchMatchBranchResult
from prediction.multi_scale_dlf import MultiScaleDLF
from prediction.se_u_net import build_se_unet_input


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


def resize_offsets_to_image_grid(offsets: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    single_image = offsets.dim() == 3
    if single_image:
        offsets = offsets.unsqueeze(0)

    source_h, source_w = offsets.shape[-2:]
    target_h, target_w = target_size
    if (source_h, source_w) == (target_h, target_w):
        return offsets.squeeze(0) if single_image else offsets

    resized = F.interpolate(offsets, size=target_size, mode="bilinear", align_corners=True)
    scale_x = float(max(target_w - 1, 1)) / float(max(source_w - 1, 1))
    scale_y = float(max(target_h - 1, 1)) / float(max(source_h - 1, 1))
    resized[:, 0] = resized[:, 0] * scale_x
    resized[:, 1] = resized[:, 1] * scale_y
    return resized.squeeze(0) if single_image else resized


def resize_topk_offsets_to_image_grid(offsets: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    single_image = offsets.dim() == 4
    if single_image:
        offsets = offsets.unsqueeze(0)

    batch_size, topk, _, source_h, source_w = offsets.shape
    target_h, target_w = target_size
    if (source_h, source_w) == (target_h, target_w):
        return offsets.squeeze(0) if single_image else offsets

    flat_offsets = offsets.reshape(batch_size * topk, 2, source_h, source_w)
    resized = resize_offsets_to_image_grid(flat_offsets, target_size)
    resized = resized.reshape(batch_size, topk, 2, target_h, target_w)
    return resized.squeeze(0) if single_image else resized


def resize_dense_maps(maps: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    original_dim = maps.dim()
    if original_dim == 2:
        maps = maps.unsqueeze(0).unsqueeze(0)
    elif original_dim == 3:
        maps = maps.unsqueeze(0)
    elif original_dim != 4:
        raise ValueError(f"Expected dense maps shape [H,W], [C,H,W], or [B,C,H,W], got {tuple(maps.shape)}")

    if maps.shape[-2:] == target_size:
        resized = maps
    else:
        resized = F.interpolate(maps.float(), size=target_size, mode="bilinear", align_corners=False)

    if original_dim == 2:
        return resized.squeeze(0).squeeze(0)
    if original_dim == 3:
        return resized.squeeze(0)
    return resized


def resize_branch_result_to_image_grid(
    branch_result: PatchMatchBranchResult,
    target_size: tuple[int, int],
) -> PatchMatchBranchResult:
    return PatchMatchBranchResult(
        offsets=resize_offsets_to_image_grid(branch_result.offsets, target_size),
        best_cost=resize_dense_maps(branch_result.best_cost, target_size),
        second_cost=resize_dense_maps(branch_result.second_cost, target_size),
        confidence=resize_dense_maps(branch_result.confidence, target_size),
        structure_map=resize_dense_maps(branch_result.structure_map, target_size),
        topk_offsets=(
            resize_topk_offsets_to_image_grid(branch_result.topk_offsets, target_size)
            if branch_result.topk_offsets is not None
            else None
        ),
        topk_costs=(
            resize_dense_maps(branch_result.topk_costs, target_size)
            if branch_result.topk_costs is not None
            else None
        ),
    )


def merge_structure_maps(*structure_maps: torch.Tensor) -> torch.Tensor:
    valid_maps = [structure_map.float() for structure_map in structure_maps if structure_map is not None]
    if not valid_maps:
        raise ValueError("Expected at least one structure map.")
    if len(valid_maps) == 1:
        return valid_maps[0]
    return torch.stack(valid_maps, dim=0).mean(dim=0)


def compute_topk_dispersion_map(*topk_offsets: torch.Tensor | None) -> torch.Tensor | None:
    dispersion_maps = []
    for offsets in topk_offsets:
        if offsets is None:
            continue
        if offsets.dim() == 4:
            offsets = offsets.unsqueeze(0)
        if offsets.dim() != 5 or offsets.shape[1] <= 1:
            continue
        dx = offsets[:, :, 0].float()
        dy = offsets[:, :, 1].float()
        dispersion = torch.sqrt(dx.var(dim=1, unbiased=False) + dy.var(dim=1, unbiased=False)).unsqueeze(1)
        dispersion_maps.append(dispersion)

    if not dispersion_maps:
        return None
    return torch.stack(dispersion_maps, dim=0).mean(dim=0)


def select_primary_feature_map(feature_maps):
    if isinstance(feature_maps, (tuple, list)):
        if not feature_maps:
            raise ValueError("Expected at least one feature map.")
        return feature_maps[len(feature_maps) // 2]
    return feature_maps


def extract_localization_inputs(
    images: torch.Tensor,
    pyramid_bb,
    pyramid_zm,
    feature_backbone: str,
    cnn_backbone: str,
    separate_transforms: bool,
    cnn_feature_norm: bool,
    pm_random_window: int,
    pm_iters: int,
    pm_beta: float,
    pm_hard_selection: bool,
    pm_use_non_local: bool,
    pm_non_local_limit: float,
    pm_reduced_precision: bool = True,
    dino_match_native_resolution: bool = False,
    localization_resolution: str = "image",
    dlf_error_scaling: str = "log1p",
    collect_stats: bool = False,
    train_feature_backbone: bool = False,
    pm_flat_threshold: float = 0.15,
    pm_margin_threshold: float = 0.10,
    pm_topk: int = 1,
):
    images_backbone = images
    if separate_transforms and feature_backbone == "cnn" and cnn_backbone == "pretrained":
        images_backbone = imagenet_normalize_tensor(images)

    device = images.device
    localization_stats = None
    peak_memory_base = None
    if collect_stats and device.type == "cuda":
        synchronize_if_cuda(device)
        torch.cuda.reset_peak_memory_stats(device)
        peak_memory_base = torch.cuda.memory_allocated(device)

    feature_context = nullcontext() if train_feature_backbone else torch.no_grad()
    match_context = nullcontext() if train_feature_backbone else torch.no_grad()

    if collect_stats:
        synchronize_if_cuda(device)
        feature_start = time.perf_counter()
    with feature_context:
        cnn_feats = pyramid_bb(images_backbone)
        if feature_backbone == "cnn" and cnn_backbone == "pretrained" and cnn_feature_norm:
            cnn_feats = tuple(F.normalize(feature, p=2, dim=1) for feature in cnn_feats)
    with torch.no_grad():
        zernike_feats = tuple(feature.detach() for feature in pyramid_zm(images))
    if collect_stats:
        synchronize_if_cuda(device)
        feature_time = time.perf_counter() - feature_start

    patchmatch_images = images
    if feature_backbone in ("dino", "dino_single") and dino_match_native_resolution:
        dino_match_size = select_primary_feature_map(cnn_feats).shape[-2:]
        if dino_match_size != images.shape[-2:]:
            patchmatch_images = F.interpolate(images, size=dino_match_size, mode="bilinear", align_corners=False)

    with match_context:
        if collect_stats:
            synchronize_if_cuda(device)
            propagation_start = time.perf_counter()
        propagator = PixelPropagator(
            patchmatch_images,
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
            topk=pm_topk,
        )
        dlf_images = images
        dlf_cnn_branch = batch_cnn_branch
        dlf_zernike_branch = batch_zernike_branch
        if localization_resolution == "image":
            if patchmatch_images.shape[-2:] != images.shape[-2:]:
                dlf_cnn_branch = resize_branch_result_to_image_grid(batch_cnn_branch, images.shape[-2:])
                dlf_zernike_branch = resize_branch_result_to_image_grid(batch_zernike_branch, images.shape[-2:])
        elif localization_resolution == "feature_grid":
            dlf_images = patchmatch_images
        else:
            raise ValueError(f"Unsupported localization resolution: {localization_resolution}")
        if collect_stats:
            synchronize_if_cuda(device)
            propagation_time = time.perf_counter() - propagation_start

        if collect_stats:
            synchronize_if_cuda(device)
            dlf_start = time.perf_counter()
        errors = MultiScaleDLF(
            dlf_images,
            dlf_cnn_branch.offsets,
            zernike_offsets=dlf_zernike_branch.offsets,
            cnn_topk_offsets=dlf_cnn_branch.topk_offsets,
            zernike_topk_offsets=dlf_zernike_branch.topk_offsets,
        ).compute_errors()
        errors = normalize_dlf_error_maps(errors, mode=dlf_error_scaling)
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

    return errors, dlf_cnn_branch, dlf_zernike_branch, dlf_images, localization_stats


def decode_and_refine_masks(
    images: torch.Tensor,
    errors: torch.Tensor,
    cnn_branch_result: PatchMatchBranchResult,
    zernike_branch_result: PatchMatchBranchResult,
    dlf_decoder,
    se_model,
    localization_images: torch.Tensor | None = None,
    output_size: tuple[int, int] | None = None,
):
    if localization_images is None:
        localization_images = images

    structure_map = merge_structure_maps(cnn_branch_result.structure_map, zernike_branch_result.structure_map)
    topk_dispersion = compute_topk_dispersion_map(
        cnn_branch_result.topk_offsets,
        zernike_branch_result.topk_offsets,
    )
    dlf_decoder_input = DLFDecoderInput(
        cross_scale_errors=errors,
        cnn_offsets=cnn_branch_result.offsets,
        zernike_offsets=zernike_branch_result.offsets,
        cnn_confidence=cnn_branch_result.confidence,
        zernike_confidence=zernike_branch_result.confidence,
        structure_map=structure_map,
        topk_dispersion=topk_dispersion,
    )

    dlf_map = dlf_decoder(dlf_decoder_input)
    se_input = build_se_unet_input(localization_images)
    target_map = se_model(se_input)
    refined_mask = torch.maximum(dlf_map, target_map)

    if output_size is None:
        output_size = images.shape[-2:]
    if refined_mask.shape[-2:] != output_size:
        refined_mask = F.interpolate(refined_mask, size=output_size, mode="bilinear", align_corners=False)
        target_map = F.interpolate(target_map, size=output_size, mode="bilinear", align_corners=False)
        dlf_map = F.interpolate(dlf_map, size=output_size, mode="bilinear", align_corners=False)
    return refined_mask, target_map, dlf_map
