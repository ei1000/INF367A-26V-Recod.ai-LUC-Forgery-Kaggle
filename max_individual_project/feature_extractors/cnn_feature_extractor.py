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
Pretrained backbone with a small spatial footprint and a 32-dim projection.
Intended as a drop-in alternative to the random-weight CNN.
'''
class PretrainedBackboneExtractor(nn.Module):
    def __init__(self, model_name="vgg16_bn", out_dim=32, freeze=True):
        super().__init__()
        self._features_frozen = False
        self.model_name = model_name

        if model_name == "vgg16_bn":
            from torchvision.models import vgg16_bn, VGG16_BN_Weights
            weights = VGG16_BN_Weights.DEFAULT
            model = vgg16_bn(weights=weights)
            # Keep only the first conv block (no pooling) to preserve resolution.
            layers = []
            for layer in model.features:
                if isinstance(layer, nn.MaxPool2d):
                    break
                layers.append(layer)
            self.features = nn.Sequential(*layers)
            in_channels = 64
        elif model_name == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
            self.features = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
            )
            in_channels = 64
        else:
            raise ValueError(f"Unsupported pretrained model: {model_name}")

        self.proj = nn.Conv2d(in_channels, out_dim, kernel_size=1, bias=False)

        if freeze:
            self.freeze_features()

    def freeze_features(self):
        self._features_frozen = True
        self.features.eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_features(self):
        self._features_frozen = False
        for param in self.features.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self._features_frozen:
            self.features.eval()
        return self

    def forward(self, x):
        x = self.features(x)
        return self.proj(x)

'''
Pyramid feature extractor using the same backbone (shared weights) to 
run the same backbone on a few different image scalings
'''
class PyramidFeatureExtractor(nn.Module):
    def __init__(self, backbone=None, rb=0.75, ru=1.5):
        super().__init__()
        self.backbone = backbone if backbone is not None else BackboneExtractor()
        self.rb = rb
        self.ru = ru

    def forward(self, Io):
        H, W = Io.shape[-2:]

        Ib = F.interpolate(Io, scale_factor=self.rb, mode='bilinear', align_corners=True)
        Iu = F.interpolate(Io, scale_factor=self.ru, mode='bilinear', align_corners=True)

        Fb = self.backbone(Ib)
        Fo = self.backbone(Io)
        Fu = self.backbone(Iu)

        # resize features back to original size
        Fb = F.interpolate(Fb, size=(H, W), mode='bilinear', align_corners=True)
        Fu = F.interpolate(Fu, size=(H, W), mode='bilinear', align_corners=True)

        return Fb, Fo, Fu
