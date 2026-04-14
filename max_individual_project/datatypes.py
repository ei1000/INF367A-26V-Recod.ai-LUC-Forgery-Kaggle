from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PatchMatchBranchResult:
    """PatchMatch outputs for one descriptor branch."""

    offsets: torch.Tensor
    best_cost: torch.Tensor
    second_cost: torch.Tensor
    confidence: torch.Tensor
    structure_map: torch.Tensor
    topk_offsets: torch.Tensor | None = None
    topk_costs: torch.Tensor | None = None


@dataclass
class DLFDecoderInput:
    """Container for batched DLF decoder inputs."""

    cross_scale_errors: torch.Tensor
    cnn_offsets: torch.Tensor
    zernike_offsets: torch.Tensor
    cnn_confidence: torch.Tensor
    zernike_confidence: torch.Tensor
    structure_map: torch.Tensor
    topk_dispersion: torch.Tensor | None = None
