import numpy as np
import torch
import torch.nn as nn


'''
Convolution block with convolution, batchnorm and relu
'''
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

'''
Full backbone utilizing multiple ConvBlocks
'''
class Backbone(nn.Module):
    def __init__(self, in_dim, out_dim, block, n_block):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 256),
        ])
    
    def forward(self, x):
        return self.blocks(x)