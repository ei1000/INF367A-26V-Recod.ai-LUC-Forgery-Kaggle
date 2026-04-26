from __future__ import annotations

import torch

from feature_extractors.cnn_feature_extractor import PretrainedBackboneExtractor, SingleScaleFeatureExtractor
from feature_extractors.zernike_feature_extractor import PyramidZernikeExtractor, default_pq_list
from prediction.dlfdecoder import DLFDecoder
from prediction.pixelmaputil_mask import post_process_mask_batch
from prediction.se_u_net import SEUNet


def post_process_predictions(
    refined_mask,
    util,
    *,
    do_post_process: bool,
    post_process_threshold: float,
    post_process_confident_threshold,
    post_process_min_component_area: int,
    post_process_smooth_probabilities: bool,
    post_process_fill_holes: bool,
    post_process_apply_closing: bool,
):
    mask_probs = refined_mask.squeeze(1)
    if do_post_process and util is not None:
        return post_process_mask_batch(
            mask_probs,
            util,
            threshold=post_process_threshold,
            confident_threshold=post_process_confident_threshold,
            min_component_area=post_process_min_component_area,
            smooth_probabilities=post_process_smooth_probabilities,
            fill_holes=post_process_fill_holes,
            apply_closing=post_process_apply_closing,
        )
    return (mask_probs >= 0.5).long()


def build_patchmatch_feature_branch(device: torch.device):
    pm_backbone = SingleScaleFeatureExtractor(
        backbone=PretrainedBackboneExtractor(
            model_name="resnet18",
            out_dim=32,
            freeze=True,
        ),
        upsample_to_input=True,
    ).to(device)
    pyramid_zm = PyramidZernikeExtractor(default_pq_list(max_order=5), kernel_size=13).to(device)
    pm_backbone.eval()
    pyramid_zm.eval()
    return pm_backbone, pyramid_zm


def build_seunet_feature_branch(
    device: torch.device,
    *,
    feature_backbone: str,
    dino_model_name: str,
    separate_transforms: bool,
    use_dino_transform: bool,
):
    from feature_extractors.dino_feature_extractor import PyramidDinoFeatureExtractor, SingleScaleDinoFeatureExtractor

    dino_extractor_cls = PyramidDinoFeatureExtractor if feature_backbone == "dino" else SingleScaleDinoFeatureExtractor
    dino_extractor = dino_extractor_cls(
        model_name=dino_model_name,
        freeze=True,
        finetune_blocks=0,
        normalize_input=True if separate_transforms else not use_dino_transform,
        proj_dim=None,
        upsample_to_input=False,
    ).to(device)
    dino_extractor.eval()
    return dino_extractor


def build_patchmatch_head(cnn_errors: torch.Tensor, device: torch.device):
    return DLFDecoder(num_error_maps=cnn_errors.shape[1]).to(device)


def build_seunet_head(dino_features: torch.Tensor, device: torch.device):
    return SEUNet(
        in_channels=dino_features.shape[1],
        out_channels=1,
        final_activation="sigmoid",
    ).to(device)


def build_localization_optimizer(dlf_decoder, se_model, learning_rate: float):
    return torch.optim.Adam(
        [
            {"params": list(dlf_decoder.parameters()), "lr": learning_rate, "name": "dlf_decoder"},
            {"params": list(se_model.parameters()), "lr": learning_rate, "name": "se_model"},
        ],
        lr=learning_rate,
    )


def set_frozen_feature_branch_modes(pm_backbone, pyramid_zm, dino_extractor):
    pm_backbone.eval()
    pyramid_zm.eval()
    dino_extractor.eval()


def set_trainable_head_modes(dlf_decoder, se_model, *, training: bool):
    if dlf_decoder is None or se_model is None:
        return
    if training:
        dlf_decoder.train()
        se_model.train()
        return
    dlf_decoder.eval()
    se_model.eval()
