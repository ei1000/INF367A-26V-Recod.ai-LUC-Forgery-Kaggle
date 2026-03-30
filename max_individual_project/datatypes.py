from dataclasses import dataclass
import torch

@dataclass
class DLFDecoderInput:
    """Container for batched DLF decoder inputs."""

    cross_scale_errors: torch.Tensor
    cnn_offsets: torch.Tensor
    zernike_offsets: torch.Tensor
