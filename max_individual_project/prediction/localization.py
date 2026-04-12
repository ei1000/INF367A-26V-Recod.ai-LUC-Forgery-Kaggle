from __future__ import annotations

import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from cross_scale_patternmatch.pixel_propagator import PixelPropagator
from dataset import imagenet_normalize_tensor
from datatypes import DLFDecoderInput
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
    dlf_error_scaling: str = "log1p",
    collect_stats: bool = False,
    train_feature_backbone: bool = False,
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
    if feature_backbone == "dino" and dino_match_native_resolution:
        dino_match_size = cnn_feats[1].shape[-2:]
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
        batch_cnn_offsets, batch_zernike_offsets = propagator.propagation_layer(
            iters=pm_iters,
            beta=pm_beta,
            hard_selection=pm_hard_selection,
            use_non_local=pm_use_non_local,
            non_local_limit=pm_non_local_limit,
        )
        if patchmatch_images.shape[-2:] != images.shape[-2:]:
            batch_cnn_offsets = resize_offsets_to_image_grid(batch_cnn_offsets, images.shape[-2:])
            batch_zernike_offsets = resize_offsets_to_image_grid(batch_zernike_offsets, images.shape[-2:])
        if collect_stats:
            synchronize_if_cuda(device)
            propagation_time = time.perf_counter() - propagation_start

        if collect_stats:
            synchronize_if_cuda(device)
            dlf_start = time.perf_counter()
        errors = MultiScaleDLF(images, batch_cnn_offsets, zernike_offsets=batch_zernike_offsets).compute_errors()
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

    return errors, batch_cnn_offsets, batch_zernike_offsets, localization_stats


def decode_and_refine_masks(
    images: torch.Tensor,
    errors: torch.Tensor,
    batch_cnn_offsets: torch.Tensor,
    batch_zernike_offsets: torch.Tensor,
    dlf_decoder,
    se_model,
):
    dlf_decoder_input = DLFDecoderInput(
        cross_scale_errors=errors,
        cnn_offsets=batch_cnn_offsets,
        zernike_offsets=batch_zernike_offsets,
    )

    dlf_map = dlf_decoder(dlf_decoder_input)
    se_input = build_se_unet_input(images)
    target_map = se_model(se_input)
    refined_mask = torch.maximum(dlf_map, target_map)
    return refined_mask, target_map, dlf_map
