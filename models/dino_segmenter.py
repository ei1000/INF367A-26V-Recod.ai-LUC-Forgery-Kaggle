import math
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoTinyDecoder(nn.Module):
    def __init__(self, in_ch: int = 768, out_ch: int = 1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, 384, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(384, 192, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(192, 96, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Conv2d(96, out_ch, 1)

    def forward(self, features: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(self.block1(features), scale_factor=2.0, mode="bilinear", align_corners=False)
        x = F.interpolate(self.block2(x), scale_factor=2.0, mode="bilinear", align_corners=False)
        x = F.interpolate(self.block3(x), scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.conv_out(x)
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)


class DinoSegmenter(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 768,
        out_ch: int = 1,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = DinoTinyDecoder(in_ch=embed_dim, out_ch=out_ch)
        self._encoder_frozen = False
        if freeze_encoder:
            self.freeze_encoder()

    @classmethod
    def from_official(
        cls,
        model_name: str = "dinov2_vitb14",
        embed_dim: int = 768,
        out_ch: int = 1,
        freeze_encoder: bool = True,
        repo: str = "facebookresearch/dinov2",
    ) -> "DinoSegmenter":
        warnings.filterwarnings(
            "ignore",
            message="xFormers is not available.*",
            category=UserWarning,
        )
        encoder = torch.hub.load(repo_or_dir=repo, model=model_name)
        return cls(
            encoder=encoder,
            embed_dim=embed_dim,
            out_ch=out_ch,
            freeze_encoder=freeze_encoder,
        )

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
                raise ValueError("Unsupported encoder output format for DINO segmenter.")

        # Drop CLS token if present and map token sequence back to feature grid.
        side = int(math.sqrt(grid_tokens.shape[1]))
        if side * side != grid_tokens.shape[1]:
            raise ValueError(
                f"Token count {grid_tokens.shape[1]} is not a perfect square; "
                "cannot reshape to spatial map."
            )

        return grid_tokens.permute(0, 2, 1).reshape(x.shape[0], grid_tokens.shape[2], side, side)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        x_pad, (pad_h, pad_w) = self._pad_to_patch_multiple(x)
        features = self.forward_features(x_pad)
        logits = self.decoder(features, target_size=(x_pad.shape[-2], x_pad.shape[-1]))
        if pad_h == 0 and pad_w == 0:
            return logits
        return logits[:, :, :orig_h, :orig_w]
