import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_batched_image(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError(f"Expected image shape [C,H,W] or [B,C,H,W], got {tuple(image.shape)}")
    return image


def _as_batched_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        if mask.shape[0] == 1:
            mask = mask.unsqueeze(0)
        else:
            mask = mask.unsqueeze(1)
    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask shape [H,W], [1,H,W], [B,H,W], or [B,1,H,W], got {tuple(mask.shape)}")
    return mask


def build_se_unet_input(image: torch.Tensor, dlf_map: torch.Tensor | None = None) -> torch.Tensor:
    image = _as_batched_image(image)
    if dlf_map is None:
        return image

    dlf_map = _as_batched_mask(dlf_map).to(device=image.device, dtype=image.dtype)
    if image.shape[0] != dlf_map.shape[0]:
        raise ValueError(f"Batch size mismatch between image {tuple(image.shape)} and DLF map {tuple(dlf_map.shape)}")
    if image.shape[-2:] != dlf_map.shape[-2:]:
        raise ValueError(f"Spatial size mismatch between image {tuple(image.shape)} and DLF map {tuple(dlf_map.shape)}")

    return torch.cat((image, dlf_map), dim=1)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class SEConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = SqueezeExcitation(out_channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.layers(x))


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16, downsample: bool = True):
        super().__init__()
        self.pool = nn.MaxPool2d(2) if downsample else nn.Identity()
        self.block = SEConvBlock(in_channels, out_channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, reduction: int = 16):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = SEConvBlock(out_channels + skip_channels, out_channels, reduction=reduction)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat((x, skip), dim=1)
        return self.block(x)


class SEUNet(nn.Module):
    """
    Scaffold for a copy-move refinement head.

    Typical usage:
    - `SEUNet(in_channels=4, out_channels=2)` for `RGB + DLF probability/logit`
    - `SEUNet(in_channels=3)` for image-only experiments
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 2,
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
        x = _as_batched_image(x)

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
