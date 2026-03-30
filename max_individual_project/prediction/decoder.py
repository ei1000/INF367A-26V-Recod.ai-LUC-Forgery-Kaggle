import torch
import torch.nn as nn
from datatypes import DLFDecoderInput

# TODO: This convblock is used both here and for feature extractors. Refactor to make more clean -
# maybe a shared modules dir?
class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.block(x)

class DLFDecoder(nn.Module):
    def __init__(self, num_error_maps):
        super().__init__()

        in_channels = num_error_maps + 4

        self.blocks = nn.ModuleList([
            ConvBlock(in_channels, 64), # input size is errors + offset map x and y's
            ConvBlock(64, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
        ])

        self.final_conv = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, input: DLFDecoderInput):
        cross_scale_errors = [
            input.cross_scale_errors[i]
            for i in range(input.cross_scale_errors.shape[0])
        ]

        tensors = cross_scale_errors + [input.cnn_offsets.unsqueeze(0), input.zernike_offsets.unsqueeze(0)]
        x = torch.cat(tensors, dim=1)

        for block in self.blocks:
            x = block(x)

        return torch.sigmoid(self.final_conv(x))

