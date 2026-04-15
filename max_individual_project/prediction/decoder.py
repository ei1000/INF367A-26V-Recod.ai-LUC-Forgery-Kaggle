from __future__ import annotations

import torch
import torch.nn as nn

from datatypes import DLFDecoderInput
from model_components.blocks import ConvBNReLU

ConvBlock = ConvBNReLU


class DLFDecoder(nn.Module):
    """Decode branch-wise DLF errors and offsets into a dense tampering prior."""

    def __init__(self, num_error_maps: int):
        super().__init__()

        self.num_error_maps = int(num_error_maps)
        in_channels = (2 * self.num_error_maps) + 4

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
            raise ValueError(f"Expected error maps shape [S,H,W] or [B,S,H,W], got {tuple(errors.shape)}")
        return errors

    def _as_batched_offsets(self, offsets: torch.Tensor) -> torch.Tensor:
        if offsets.dim() == 3:
            offsets = offsets.unsqueeze(0)
        if offsets.dim() != 4 or offsets.shape[1] != 2:
            raise ValueError(f"Expected offsets shape [2,H,W] or [B,2,H,W], got {tuple(offsets.shape)}")
        return offsets

    def normalize_offsets(self, offsets: torch.Tensor) -> torch.Tensor:
        _, _, height, width = offsets.shape
        scale_x = max(width - 1, 1)
        scale_y = max(height - 1, 1)

        normalized = offsets.to(dtype=torch.float32).clone()
        normalized[:, 0] = normalized[:, 0] / float(scale_x)
        normalized[:, 1] = normalized[:, 1] / float(scale_y)
        return normalized

    def forward(self, input: DLFDecoderInput):
        device = self.final_conv.weight.device
        cnn_error_maps = self._as_batched_errors(input.cnn_error_maps).to(device)
        zernike_error_maps = self._as_batched_errors(input.zernike_error_maps).to(device)
        cnn_offsets = self.normalize_offsets(self._as_batched_offsets(input.cnn_offsets)).to(device)
        zernike_offsets = self.normalize_offsets(self._as_batched_offsets(input.zernike_offsets)).to(device)

        if cnn_error_maps.shape[0] != zernike_error_maps.shape[0]:
            raise ValueError(
                f"Batch size mismatch between CNN and Zernike error maps: "
                f"{tuple(cnn_error_maps.shape)} vs {tuple(zernike_error_maps.shape)}"
            )
        if cnn_error_maps.shape[1] != self.num_error_maps or zernike_error_maps.shape[1] != self.num_error_maps:
            raise ValueError(
                f"Expected {self.num_error_maps} error maps per branch, got "
                f"{tuple(cnn_error_maps.shape)} and {tuple(zernike_error_maps.shape)}"
            )
        if cnn_error_maps.shape[0] != cnn_offsets.shape[0]:
            raise ValueError(
                f"Batch size mismatch between error maps {tuple(cnn_error_maps.shape)} and offsets {tuple(cnn_offsets.shape)}"
            )
        if cnn_offsets.shape != zernike_offsets.shape:
            raise ValueError(
                f"Offset tensors must have the same shape, got {tuple(cnn_offsets.shape)} and {tuple(zernike_offsets.shape)}"
            )
        if cnn_error_maps.shape[-2:] != cnn_offsets.shape[-2:]:
            raise ValueError(
                f"Spatial size mismatch between CNN errors {tuple(cnn_error_maps.shape)} and offsets {tuple(cnn_offsets.shape)}"
            )
        if zernike_error_maps.shape[-2:] != zernike_offsets.shape[-2:]:
            raise ValueError(
                f"Spatial size mismatch between Zernike errors {tuple(zernike_error_maps.shape)} "
                f"and offsets {tuple(zernike_offsets.shape)}"
            )
        if cnn_error_maps.shape[-2:] != zernike_error_maps.shape[-2:]:
            raise ValueError(
                f"Spatial size mismatch between CNN and Zernike error maps: "
                f"{tuple(cnn_error_maps.shape)} vs {tuple(zernike_error_maps.shape)}"
            )

        inputs = [cnn_error_maps, zernike_error_maps, cnn_offsets, zernike_offsets]

        x = torch.cat(inputs, dim=1)

        for block in self.blocks:
            x = block(x)

        return torch.sigmoid(self.final_conv(x))
