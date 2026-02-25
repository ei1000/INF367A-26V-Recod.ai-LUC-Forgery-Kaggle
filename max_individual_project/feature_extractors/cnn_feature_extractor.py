import torch.nn as nn
import torch.nn.functional as F

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
class BackboneExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 256),
        ])

        self.final_conv = nn.Conv2d(256, 32, kernel_size=1)  # c = 32
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return self.final_conv(x)

'''
Pyramid feature extractor using the same backbone (shared weights) to 
run the same backbone on a few different image scalings
'''
class PyramidFeatureExtractor(nn.Module):
    def __init__(self, rb=0.75, ru=1.5):
        super().__init__()
        self.backbone = BackboneExtractor()
        self.rb = rb
        self.ru = ru

    def forward(self, Io):
        H, W = Io.shape[-2:]

        Ib = F.interpolate(Io, scale_factor=self.rb, mode='bilinear')
        Iu = F.interpolate(Io, scale_factor=self.ru, mode='bilinear')

        Fb = self.backbone(Ib)
        Fo = self.backbone(Io)
        Fu = self.backbone(Iu)

        # resize features back to original size
        Fb = F.interpolate(Fb, size=(H, W), mode='bilinear')
        Fu = F.interpolate(Fu, size=(H, W), mode='bilinear')

        return Fb, Fo, Fu
