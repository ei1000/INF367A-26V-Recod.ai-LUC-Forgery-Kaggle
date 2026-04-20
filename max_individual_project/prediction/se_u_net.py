import torch
import torch.nn as nn

try:
    from ..model_components.se_blocks import DecoderStage, EncoderStage, SEConvBlock, SqueezeExcitation
except ImportError:
    from model_components.se_blocks import DecoderStage, EncoderStage, SEConvBlock, SqueezeExcitation

__all__ = [
    "SqueezeExcitation",
    "SEConvBlock",
    "EncoderStage",
    "DecoderStage",
    "SEUNet",
    "as_batched_image",
]


def as_batched_image(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError(f"Expected image shape [C,H,W] or [B,C,H,W], got {tuple(image.shape)}")
    return image


class SEUNet(nn.Module):
    """
    Scaffold for a copy-move refinement head.

    Typical usage:
    - `SEUNet(in_channels=3, out_channels=1, final_activation="sigmoid")` for image-only refinement
    - `SEUNet(in_channels=dino_channels, ...)` for frozen DINO feature refinement
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        encoder_channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        bottleneck_channels: int = 512,
        se_reduction: int = 16,
        final_activation: str | None = None,
    ):
        super().__init__()

        c1, c2, c3, c4 = encoder_channels
        self.final_activation = final_activation

        self.enc1 = EncoderStage(in_channels, c1, reduction=se_reduction, downsample=False)
        self.enc2 = EncoderStage(c1, c2, reduction=se_reduction)
        self.enc3 = EncoderStage(c2, c3, reduction=se_reduction)
        self.enc4 = EncoderStage(c3, c4, reduction=se_reduction)

        self.bottleneck = EncoderStage(c4, bottleneck_channels, reduction=se_reduction)

        self.dec4 = DecoderStage(bottleneck_channels, c4, c4, reduction=se_reduction)
        self.dec3 = DecoderStage(c4, c3, c3, reduction=se_reduction)
        self.dec2 = DecoderStage(c3, c2, c2, reduction=se_reduction)
        self.dec1 = DecoderStage(c2, c1, c1, reduction=se_reduction)

        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = as_batched_image(x)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        bottleneck = self.bottleneck(e4)

        d4 = self.dec4(bottleneck, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        logits = self.head(d1)
        if self.final_activation == "sigmoid":
            return torch.sigmoid(logits)
        if self.final_activation == "softmax":
            return torch.softmax(logits, dim=1)
        return logits
