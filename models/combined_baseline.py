from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from max_individual_project.datatypes import DLFDecoderInput
from max_individual_project.pipeline_helpers import build_patchmatch_feature_branch, build_patchmatch_head
from max_individual_project.prediction.localization import extract_localization_inputs
from max_individual_project.training.checkpointing import load_module_state
from models.dino_segmenter import DinoSegmenter


@dataclass
class CombinedForwardOutputs:
    combined_logits: torch.Tensor
    combined_prob: torch.Tensor
    baseline_logits: torch.Tensor
    baseline_prob: torch.Tensor
    patchmatch_prob: torch.Tensor
    localization_stats: dict[str, float] | None = None


class CombinedBaselineModel(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dino_model_name: str,
        dino_embed_dim: int,
        freeze_dino_encoder: bool,
        dino_mean: tuple[float, float, float],
        dino_std: tuple[float, float, float],
        separate_transforms: bool,
        cnn_feature_norm: bool,
        pm_random_window: int,
        pm_iters: int,
        pm_beta: float,
        pm_hard_selection: bool,
        pm_use_non_local: bool,
        pm_non_local_limit: float,
        pm_flat_threshold: float,
        pm_margin_threshold: float,
        pm_topk: int,
        pm_reduced_precision: bool,
        dlf_error_scaling: str = "log1p",
    ):
        super().__init__()
        self.runtime_device = device
        self.separate_transforms = separate_transforms
        self.cnn_feature_norm = cnn_feature_norm
        self.pm_random_window = pm_random_window
        self.pm_iters = pm_iters
        self.pm_beta = pm_beta
        self.pm_hard_selection = pm_hard_selection
        self.pm_use_non_local = pm_use_non_local
        self.pm_non_local_limit = pm_non_local_limit
        self.pm_flat_threshold = pm_flat_threshold
        self.pm_margin_threshold = pm_margin_threshold
        self.pm_topk = pm_topk
        self.pm_reduced_precision = pm_reduced_precision
        self.dlf_error_scaling = dlf_error_scaling

        self.baseline_model = DinoSegmenter.from_official(
            model_name=dino_model_name,
            embed_dim=dino_embed_dim,
            freeze_encoder=freeze_dino_encoder,
        )
        self.pm_backbone, self.pyramid_zm = build_patchmatch_feature_branch(device)
        self.patchmatch_decoder: nn.Module | None = None
        self._pending_patchmatch_decoder_state: dict[str, torch.Tensor] | None = None
        self.patchmatch_decoder_restored_fully = True

        self.register_buffer("dino_mean", torch.tensor(dino_mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("dino_std", torch.tensor(dino_std, dtype=torch.float32).view(1, 3, 1, 1))
        self.to(device)

    def train(self, mode: bool = True):
        super().train(mode)
        self.pm_backbone.eval()
        self.pyramid_zm.eval()
        if self.patchmatch_decoder is not None:
            self.patchmatch_decoder.train(mode)
        return self

    def _normalize_for_dino(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.dino_mean) / self.dino_std

    def set_pending_patchmatch_decoder_state(self, state_dict: dict[str, torch.Tensor] | None) -> None:
        self._pending_patchmatch_decoder_state = state_dict
        self.patchmatch_decoder_restored_fully = state_dict is None

    def _ensure_patchmatch_decoder(self, cnn_errors: torch.Tensor) -> None:
        if self.patchmatch_decoder is None:
            self.patchmatch_decoder = build_patchmatch_head(cnn_errors, cnn_errors.device)
            if self._pending_patchmatch_decoder_state is not None:
                self.patchmatch_decoder_restored_fully = load_module_state(
                    self.patchmatch_decoder,
                    self._pending_patchmatch_decoder_state,
                    "patchmatch_decoder",
                )
                self._pending_patchmatch_decoder_state = None
            else:
                self.patchmatch_decoder_restored_fully = True

    def initialize_patchmatch_decoder(self, images: torch.Tensor) -> None:
        with torch.no_grad():
            self.compute_outputs(images)

    def patchmatch_decoder_state_dict(self) -> dict[str, torch.Tensor] | None:
        if self.patchmatch_decoder is None:
            return None
        return self.patchmatch_decoder.state_dict()

    def compute_outputs(
        self,
        images: torch.Tensor,
        *,
        collect_patchmatch_stats: bool = False,
    ) -> CombinedForwardOutputs:
        baseline_logits = self.baseline_model(self._normalize_for_dino(images))
        baseline_prob = torch.sigmoid(baseline_logits)

        cnn_errors, zernike_errors, cnn_branch_result, zernike_branch_result, _dino_features, localization_stats = extract_localization_inputs(
            images=images,
            pm_backbone=self.pm_backbone,
            pyramid_zm=self.pyramid_zm,
            dino_extractor=None,
            separate_transforms=self.separate_transforms,
            cnn_feature_norm=self.cnn_feature_norm,
            pm_random_window=self.pm_random_window,
            pm_iters=self.pm_iters,
            pm_beta=self.pm_beta,
            pm_hard_selection=self.pm_hard_selection,
            pm_use_non_local=self.pm_use_non_local,
            pm_non_local_limit=self.pm_non_local_limit,
            pm_reduced_precision=self.pm_reduced_precision,
            localization_resolution="image",
            dlf_error_scaling=self.dlf_error_scaling,
            collect_stats=collect_patchmatch_stats,
            pm_flat_threshold=self.pm_flat_threshold,
            pm_margin_threshold=self.pm_margin_threshold,
            pm_topk=self.pm_topk,
        )

        self._ensure_patchmatch_decoder(cnn_errors)
        if self.patchmatch_decoder is None:
            raise RuntimeError("PatchMatch decoder failed to initialize.")

        patchmatch_prob = self.patchmatch_decoder(
            DLFDecoderInput(
                cnn_error_maps=cnn_errors,
                zernike_error_maps=zernike_errors,
                cnn_offsets=cnn_branch_result.offsets,
                zernike_offsets=zernike_branch_result.offsets,
            )
        )
        if patchmatch_prob.shape[-2:] != baseline_prob.shape[-2:]:
            patchmatch_prob = F.interpolate(
                patchmatch_prob,
                size=baseline_prob.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        combined_prob = torch.maximum(baseline_prob, patchmatch_prob)
        combined_logits = torch.logit(combined_prob.clamp(min=1e-6, max=1.0 - 1e-6))
        return CombinedForwardOutputs(
            combined_logits=combined_logits,
            combined_prob=combined_prob,
            baseline_logits=baseline_logits,
            baseline_prob=baseline_prob,
            patchmatch_prob=patchmatch_prob,
            localization_stats=localization_stats,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.compute_outputs(images).combined_logits
