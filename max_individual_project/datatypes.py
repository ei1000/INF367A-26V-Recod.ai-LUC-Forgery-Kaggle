from dataclasses import dataclass
import torch

@dataclass
class DLFDecoderInput:
    '''
    '''
    cross_scale_errors: list[torch.tensor]
    cnn_offsets: torch.tensor
    zernike_offsets: torch.tensor
