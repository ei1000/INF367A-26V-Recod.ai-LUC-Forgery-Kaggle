from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """Shared convolution block used by the lightweight CNN modules."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
