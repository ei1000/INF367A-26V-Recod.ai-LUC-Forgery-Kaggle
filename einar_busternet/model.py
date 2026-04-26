from __future__ import annotations

import math
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfCorrelPercPooling(nn.Module):
    """Cosine self-correlation with percentile pooling on a spatial feature grid."""

    def __init__(self, nb_pools: int = 100, eps: float = 1e-6) -> None:
        super().__init__()
        if nb_pools <= 0:
            raise ValueError(f"nb_pools must be positive, got {nb_pools}")
        self.nb_pools = int(nb_pools)
        self.eps = float(eps)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"Expected features with shape (B,C,H,W), got {tuple(features.shape)}")

        b, c, h, w = features.shape
        locations = h * w
        flat = features.reshape(b, c, locations)
        norm = F.normalize(flat, dim=1, eps=self.eps)
        similarity = torch.bmm(norm.transpose(1, 2), norm)
        sorted_similarity = similarity.sort(dim=-1, descending=True).values

        pool_positions = torch.linspace(
            0,
            locations - 1,
            self.nb_pools,
            device=features.device,
        ).round().long()
        gather_index = pool_positions.view(1, 1, self.nb_pools).expand(b, locations, self.nb_pools)
        pooled = sorted_similarity.gather(dim=-1, index=gather_index)
        return pooled.permute(0, 2, 1).reshape(b, self.nb_pools, h, w)


class ManiGridDecoder(nn.Module):
    def __init__(self, in_ch: int = 768, out_ch: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_ch, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class SimiGridDecoder(nn.Module):
    def __init__(self, in_ch: int = 100, out_ch: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 96, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, out_ch, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _fusion_head(out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(226, 160, 1),
        nn.BatchNorm2d(160),
        nn.ReLU(inplace=True),
        nn.Conv2d(160, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, out_channels, 3, padding=1),
    )


class DinoBusterNet(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 768,
        nb_pools: int = 100,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.mani_decoder = ManiGridDecoder(in_ch=embed_dim, out_ch=128)
        self.mani_classifier = nn.Conv2d(128, 1, 3, padding=1)
        self.corr_pooling = SelfCorrelPercPooling(nb_pools=nb_pools)
        self.simi_decoder = SimiGridDecoder(in_ch=nb_pools, out_ch=96)
        self.simi_classifier = nn.Conv2d(96, 1, 3, padding=1)
        self.fusion = _fusion_head(out_channels=3)
        self._encoder_frozen = False
        if freeze_encoder:
            self.freeze_encoder()

    @classmethod
    def from_official(
        cls,
        model_name: str = "dinov2_vitb14",
        embed_dim: int = 768,
        nb_pools: int = 100,
        freeze_encoder: bool = True,
        repo: str = "facebookresearch/dinov2",
    ) -> "DinoBusterNet":
        warnings.filterwarnings(
            "ignore",
            message="xFormers is not available.*",
            category=UserWarning,
        )
        encoder = torch.hub.load(repo_or_dir=repo, model=model_name)
        return cls(
            encoder=encoder,
            embed_dim=embed_dim,
            nb_pools=nb_pools,
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
                raise ValueError("Unsupported encoder output format for DINO BusterNet.")

        side = int(math.sqrt(grid_tokens.shape[1]))
        if side * side != grid_tokens.shape[1]:
            raise ValueError(
                f"Token count {grid_tokens.shape[1]} is not a perfect square; "
                "cannot reshape to spatial map."
            )

        return grid_tokens.permute(0, 2, 1).reshape(x.shape[0], grid_tokens.shape[2], side, side)

    def _branch_grid_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mani_features = self.mani_decoder(features)
        simi_features = self.corr_pooling(features)
        simi_features = self.simi_decoder(simi_features)
        return mani_features, simi_features

    def _branch_grid_logits_from_features(
        self,
        mani_features: torch.Tensor,
        simi_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mani_classifier(mani_features), self.simi_classifier(simi_features)

    def _fusion_grid_from_branch_features(
        self,
        mani_features: torch.Tensor,
        simi_features: torch.Tensor,
    ) -> torch.Tensor:
        mani_grid, simi_grid = self._branch_grid_logits_from_features(mani_features, simi_features)
        return self.fusion(torch.cat([mani_features, simi_features, mani_grid, simi_grid], dim=1))

    def _upsample_and_crop(self, logits: torch.Tensor, padded_size: tuple[int, int], original_size: tuple[int, int]) -> torch.Tensor:
        logits = F.interpolate(logits, size=padded_size, mode="bilinear", align_corners=False)
        orig_h, orig_w = original_size
        return logits[:, :, :orig_h, :orig_w]

    def forward_branches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orig_size = (x.shape[-2], x.shape[-1])
        x_pad, _ = self._pad_to_patch_multiple(x)
        features = self.forward_features(x_pad)
        mani_features, simi_features = self._branch_grid_features(features)
        mani_grid, simi_grid = self._branch_grid_logits_from_features(mani_features, simi_features)
        padded_size = (x_pad.shape[-2], x_pad.shape[-1])
        return (
            self._upsample_and_crop(mani_grid, padded_size, orig_size),
            self._upsample_and_crop(simi_grid, padded_size, orig_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = (x.shape[-2], x.shape[-1])
        x_pad, _ = self._pad_to_patch_multiple(x)
        features = self.forward_features(x_pad)
        mani_features, simi_features = self._branch_grid_features(features)
        fused_grid = self._fusion_grid_from_branch_features(mani_features, simi_features)
        padded_size = (x_pad.shape[-2], x_pad.shape[-1])
        return self._upsample_and_crop(fused_grid, padded_size, orig_size)


class BinaryFusionDinoBusterNet(DinoBusterNet):
    """BusterNet variant whose fusion head predicts the binary union mask."""

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 768,
        nb_pools: int = 100,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__(
            encoder=encoder,
            embed_dim=embed_dim,
            nb_pools=nb_pools,
            freeze_encoder=freeze_encoder,
        )
        self.fusion = _fusion_head(out_channels=1)

    @classmethod
    def from_official(
        cls,
        model_name: str = "dinov2_vitb14",
        embed_dim: int = 768,
        nb_pools: int = 100,
        freeze_encoder: bool = True,
        repo: str = "facebookresearch/dinov2",
    ) -> "BinaryFusionDinoBusterNet":
        warnings.filterwarnings(
            "ignore",
            message="xFormers is not available.*",
            category=UserWarning,
        )
        encoder = torch.hub.load(repo_or_dir=repo, model=model_name)
        return cls(
            encoder=encoder,
            embed_dim=embed_dim,
            nb_pools=nb_pools,
            freeze_encoder=freeze_encoder,
        )


class BusterNetUnionWrapper(nn.Module):
    """Expose BusterNet as one-channel binary logits for baseline evaluators."""

    def __init__(self, model: nn.Module, eps: float = 1e-6) -> None:
        super().__init__()
        self.model = model
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        if logits.shape[1] == 1:
            return logits
        if logits.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 output channels, got {logits.shape[1]}")
        probs = logits.softmax(dim=1)
        forgery_prob = probs[:, 1:2] + probs[:, 2:3]
        return torch.logit(forgery_prob.clamp(self.eps, 1.0 - self.eps))
