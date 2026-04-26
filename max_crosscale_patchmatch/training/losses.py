from __future__ import annotations

import torch


def dice_loss(pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if pred_mask.dim() == 3:
        pred_mask = pred_mask.unsqueeze(1)
    if target_mask.dim() == 3:
        target_mask = target_mask.unsqueeze(1)

    pred_mask = pred_mask.float()
    target_mask = target_mask.float()

    pred_flat = pred_mask.reshape(pred_mask.shape[0], -1)
    target_flat = target_mask.reshape(target_mask.shape[0], -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1 - dice.mean()


def localization_loss_terms(
    refined_mask: torch.Tensor,
    target_map: torch.Tensor,
    dlf_map: torch.Tensor,
    target_mask: torch.Tensor,
    mprime_loss_weight: float = 0.0,
    empty_target_penalty_weight: float = 0.0,
):
    def empty_target_penalty(pred_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if pred_map.dim() == 3:
            pred_map = pred_map.unsqueeze(1)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        empty_samples = mask.flatten(start_dim=1).amax(dim=1) == 0
        if not empty_samples.any():
            return pred_map.new_tensor(0.0)

        empty_preds = pred_map[empty_samples].flatten(start_dim=1)
        return 0.5 * empty_preds.mean(dim=1).mean() + 0.5 * empty_preds.amax(dim=1).mean()

    ldfm = dice_loss(refined_mask, target_mask)
    lmrd = dice_loss(target_map, target_mask)
    mprime_loss = dice_loss(dlf_map, target_mask)
    empty_target_loss = refined_mask.new_tensor(0.0)
    empty_refined_loss = refined_mask.new_tensor(0.0)
    empty_target_map_loss = refined_mask.new_tensor(0.0)
    empty_mprime_map_loss = refined_mask.new_tensor(0.0)

    if empty_target_penalty_weight > 0.0:
        empty_refined_loss = empty_target_penalty(refined_mask, target_mask)
        empty_target_map_loss = empty_target_penalty(target_map, target_mask)
        empty_mprime_map_loss = empty_target_penalty(dlf_map, target_mask)
        empty_target_loss = torch.stack(
            (empty_refined_loss, empty_target_map_loss, empty_mprime_map_loss)
        ).mean()

    total_loss = (
        ldfm
        + lmrd
        + (mprime_loss_weight * mprime_loss)
        + (empty_target_penalty_weight * empty_target_loss)
    )
    return (
        total_loss,
        ldfm,
        lmrd,
        mprime_loss,
        empty_target_loss,
        empty_refined_loss,
        empty_target_map_loss,
        empty_mprime_map_loss,
    )


def summarize_branch_activity(dlf_map: torch.Tensor, target_map: torch.Tensor) -> dict[str, float]:
    return {
        "mprime_positive_rate": float((dlf_map >= 0.5).float().mean().item()),
        "mprime_wins_rate": float((dlf_map > target_map).float().mean().item()),
        "target_positive_rate": float((target_map >= 0.5).float().mean().item()),
    }
