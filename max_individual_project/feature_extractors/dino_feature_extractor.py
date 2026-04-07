import math
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoFeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        repo: str = "facebookresearch/dinov2",
        freeze: bool = True,
        normalize_input: bool = True,
        proj_dim: int | None = None,
    ):
        super().__init__()
        warnings.filterwarnings(
            "ignore",
            message="xFormers is not available.*",
            category=UserWarning,
        )
        self.encoder = torch.hub.load(repo_or_dir=repo, model=model_name)
        self._encoder_frozen = False
        self.normalize_input = normalize_input
        self.proj_dim = proj_dim
        self.proj = None
        if freeze:
            self.freeze_encoder()

        # ImageNet normalization for DINOv2
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def freeze_encoder(self) -> None:
        self._encoder_frozen = True
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        self._encoder_frozen = False
        for param in self.encoder.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder_frozen:
            self.encoder.eval()
        return self

    def _get_patch_size(self) -> tuple[int, int]:
        patch_size = getattr(self.encoder, "patch_size", 14)
        if isinstance(patch_size, tuple):
            return int(patch_size[0]), int(patch_size[1])
        return int(patch_size), int(patch_size)

    def _pad_to_patch_multiple(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        patch_h, patch_w = self._get_patch_size()
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (patch_h - (h % patch_h)) % patch_h
        pad_w = (patch_w - (w % patch_w)) % patch_w
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), (pad_h, pad_w)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.encoder, "forward_features"):
            feats: Any = self.encoder.forward_features(x)
            if isinstance(feats, dict):
                if "x_norm_patchtokens" in feats:
                    grid_tokens = feats["x_norm_patchtokens"]
                elif "x_prenorm" in feats:
                    tokens = feats["x_prenorm"]
                    n_tokens = tokens.shape[1]
                    grid_tokens = tokens[:, 1:, :] if int(math.sqrt(n_tokens - 1)) ** 2 == (n_tokens - 1) else tokens
                else:
                    raise ValueError("Unsupported DINOv2 forward_features dict keys.")
            elif isinstance(feats, torch.Tensor):
                grid_tokens = feats
            else:
                raise ValueError("Unsupported DINOv2 forward_features return type.")
        else:
            encoder_out: Any = self.encoder(x)
            if isinstance(encoder_out, torch.Tensor):
                n_tokens = encoder_out.shape[1]
                grid_tokens = encoder_out[:, 1:, :] if int(math.sqrt(n_tokens - 1)) ** 2 == (n_tokens - 1) else encoder_out
            else:
                raise ValueError("Unsupported encoder output format for DINO.")

        # Drop CLS token if present and map token sequence back to feature grid.
        side = int(math.sqrt(grid_tokens.shape[1]))
        if side * side != grid_tokens.shape[1]:
            raise ValueError(
                f"Token count {grid_tokens.shape[1]} is not a perfect square; "
                "cannot reshape to spatial map."
            )

        return grid_tokens.permute(0, 2, 1).reshape(x.shape[0], grid_tokens.shape[2], side, side)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            x = (x - self.mean) / self.std

        x_pad, _ = self._pad_to_patch_multiple(x)
        features = self.forward_features(x_pad)

        if self.proj_dim is not None:
            if self.proj is None:
                self.proj = nn.Conv2d(features.shape[1], self.proj_dim, kernel_size=1, bias=False).to(features.device)
            features = self.proj(features)

        return features


class PyramidDinoFeatureExtractor(nn.Module):
    def __init__(self, rb=0.75, ru=1.5, upsample_to_input: bool = True, **dino_kwargs):
        super().__init__()
        self.backbone = DinoFeatureExtractor(**dino_kwargs)
        self.rb = rb
        self.ru = ru
        self.upsample_to_input = upsample_to_input

    def forward(self, Io):
        Ib = F.interpolate(Io, scale_factor=self.rb, mode='bilinear', align_corners=False)
        Iu = F.interpolate(Io, scale_factor=self.ru, mode='bilinear', align_corners=False)

        Fb = self.backbone(Ib)
        Fo = self.backbone(Io)
        Fu = self.backbone(Iu)

        if self.upsample_to_input:
            output_size = Io.shape[-2:]
            Fb = F.interpolate(Fb, size=output_size, mode='bilinear', align_corners=False)
            Fo = F.interpolate(Fo, size=output_size, mode='bilinear', align_corners=False)
            Fu = F.interpolate(Fu, size=output_size, mode='bilinear', align_corners=False)
        else:
            output_size = Fo.shape[-2:]
            if Fb.shape[-2:] != output_size:
                Fb = F.interpolate(Fb, size=output_size, mode='bilinear', align_corners=False)
            if Fu.shape[-2:] != output_size:
                Fu = F.interpolate(Fu, size=output_size, mode='bilinear', align_corners=False)

        return Fb, Fo, Fu
