from __future__ import annotations

import torch
import torch.nn as nn

from datatypes import DLFDecoderInput
from model_components.blocks import ConvBNReLU

ConvBlock = ConvBNReLU


class DLFDecoder(nn.Module):
    def __init__(self, num_error_maps: int, include_topk_dispersion: bool = False):
        super().__init__()

        self.include_topk_dispersion = bool(include_topk_dispersion)
        in_channels = num_error_maps + 7 + int(self.include_topk_dispersion)

        self.blocks = nn.ModuleList(
            [
                ConvBlock(in_channels, 64),
                ConvBlock(64, 64),
                ConvBlock(64, 128),
                ConvBlock(128, 128),
            ]
        )

        self.final_conv = nn.Conv2d(128, 1, kernel_size=1)

    def _as_batched_errors(self, errors: torch.Tensor) -> torch.Tensor:
        if errors.dim() == 3:
            errors = errors.unsqueeze(0)
        if errors.dim() != 4:
            raise ValueError(f"Expected cross-scale errors shape [S,H,W] or [B,S,H,W], got {tuple(errors.shape)}")
        return errors

    def _as_batched_offsets(self, offsets: torch.Tensor) -> torch.Tensor:
        if offsets.dim() == 3:
            offsets = offsets.unsqueeze(0)
        if offsets.dim() != 4 or offsets.shape[1] != 2:
            raise ValueError(f"Expected offsets shape [2,H,W] or [B,2,H,W], got {tuple(offsets.shape)}")
        return offsets

    def _as_batched_single_channel(self, values: torch.Tensor, name: str) -> torch.Tensor:
        if values.dim() == 2:
            values = values.unsqueeze(0).unsqueeze(0)
        elif values.dim() == 3:
            values = values.unsqueeze(0)
        if values.dim() != 4 or values.shape[1] != 1:
            raise ValueError(f"Expected {name} shape [H,W], [1,H,W], or [B,1,H,W], got {tuple(values.shape)}")
        return values

    def normalize_offsets(self, offsets: torch.Tensor) -> torch.Tensor:
        _, _, height, width = offsets.shape
        scale_x = max(width - 1, 1)
        scale_y = max(height - 1, 1)

        normalized = offsets.to(dtype=torch.float32).clone()
        normalized[:, 0] = normalized[:, 0] / float(scale_x)
        normalized[:, 1] = normalized[:, 1] / float(scale_y)
        return normalized

    def normalize_dispersion(self, dispersion: torch.Tensor) -> torch.Tensor:
        _, _, height, width = dispersion.shape
        diagonal = max((max(width - 1, 1) ** 2 + max(height - 1, 1) ** 2) ** 0.5, 1.0)
        return dispersion.to(dtype=torch.float32) / float(diagonal)

    def forward(self, input: DLFDecoderInput):
        device = self.final_conv.weight.device
        cross_scale_errors = self._as_batched_errors(input.cross_scale_errors).to(device)
        cnn_offsets = self.normalize_offsets(self._as_batched_offsets(input.cnn_offsets)).to(device)
        zernike_offsets = self.normalize_offsets(self._as_batched_offsets(input.zernike_offsets)).to(device)
        cnn_confidence = self._as_batched_single_channel(input.cnn_confidence, "cnn_confidence").to(device, dtype=torch.float32)
        zernike_confidence = self._as_batched_single_channel(input.zernike_confidence, "zernike_confidence").to(device, dtype=torch.float32)
        structure_map = self._as_batched_single_channel(input.structure_map, "structure_map").to(device, dtype=torch.float32)

        if cross_scale_errors.shape[0] != cnn_offsets.shape[0]:
            raise ValueError(
                f"Batch size mismatch between errors {tuple(cross_scale_errors.shape)} and offsets {tuple(cnn_offsets.shape)}"
            )
        if cnn_offsets.shape != zernike_offsets.shape:
            raise ValueError(
                f"Offset tensors must have the same shape, got {tuple(cnn_offsets.shape)} and {tuple(zernike_offsets.shape)}"
            )
        if cross_scale_errors.shape[-2:] != cnn_offsets.shape[-2:]:
            raise ValueError(
                f"Spatial size mismatch between errors {tuple(cross_scale_errors.shape)} and offsets {tuple(cnn_offsets.shape)}"
            )
        if cnn_confidence.shape != zernike_confidence.shape or cnn_confidence.shape != structure_map.shape:
            raise ValueError(
                "Confidence and structure maps must all have the same shape, got "
                f"{tuple(cnn_confidence.shape)}, {tuple(zernike_confidence.shape)}, and {tuple(structure_map.shape)}"
            )
        if cnn_confidence.shape[-2:] != cross_scale_errors.shape[-2:]:
            raise ValueError(
                f"Spatial size mismatch between errors {tuple(cross_scale_errors.shape)} and confidence maps {tuple(cnn_confidence.shape)}"
            )

        inputs = [cross_scale_errors, cnn_offsets, zernike_offsets, cnn_confidence, zernike_confidence, structure_map]
        if self.include_topk_dispersion:
            if input.topk_dispersion is None:
                topk_dispersion = torch.zeros_like(structure_map, device=device)
            else:
                topk_dispersion = self.normalize_dispersion(
                    self._as_batched_single_channel(input.topk_dispersion, "topk_dispersion")
                ).to(device)
            if topk_dispersion.shape != structure_map.shape:
                raise ValueError(
                    f"topk_dispersion must match structure_map shape, got {tuple(topk_dispersion.shape)} and {tuple(structure_map.shape)}"
                )
            inputs.append(topk_dispersion)

        x = torch.cat(inputs, dim=1)

        for block in self.blocks:
            x = block(x)

        return torch.sigmoid(self.final_conv(x))
