from __future__ import annotations

import torch
from tqdm import tqdm

from max_individual_project.training.losses import dice_loss


def build_combined_optimizer(model, learning_rate: float) -> torch.optim.Optimizer:
    if model.patchmatch_decoder is None:
        raise RuntimeError("PatchMatch decoder must be initialized before building the optimizer.")

    return torch.optim.Adam(
        [
            {
                "params": [param for param in model.baseline_model.parameters() if param.requires_grad],
                "lr": learning_rate,
                "name": "baseline_model",
            },
            {
                "params": [param for param in model.patchmatch_decoder.parameters() if param.requires_grad],
                "lr": learning_rate,
                "name": "patchmatch_decoder",
            },
        ],
        lr=learning_rate,
    )


def restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    state_dict: dict | None,
    learning_rate: float,
) -> None:
    if state_dict is None:
        return
    optimizer.load_state_dict(state_dict)
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def compute_combined_loss_terms(
    outputs,
    masks: torch.Tensor,
    loss_fn: torch.nn.Module,
    patchmatch_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    combined_bce = loss_fn(outputs.combined_logits, masks)
    baseline_dice = dice_loss(outputs.baseline_prob, masks)
    patchmatch_dice = dice_loss(outputs.patchmatch_prob, masks)
    total_loss = combined_bce + baseline_dice + (patchmatch_loss_weight * patchmatch_dice)
    return total_loss, {
        "loss": total_loss,
        "combined_bce": combined_bce,
        "baseline_dice": baseline_dice,
        "patchmatch_dice": patchmatch_dice,
    }


def train_combined_one_epoch(
    model,
    train_loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    grad_clip_max_norm: float,
    epoch_idx: int,
    patchmatch_loss_weight: float,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "combined_bce": 0.0,
        "baseline_dice": 0.0,
        "patchmatch_dice": 0.0,
    }

    for imgs, masks in tqdm(train_loader, desc=f"epoch {epoch_idx + 1} train"):
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model.compute_outputs(imgs)
            loss, loss_terms = compute_combined_loss_terms(
                outputs,
                masks,
                loss_fn=loss_fn,
                patchmatch_loss_weight=patchmatch_loss_weight,
            )

        baseline_params = optimizer.param_groups[0]["params"]
        patchmatch_params = optimizer.param_groups[1]["params"]
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(baseline_params, max_norm=grad_clip_max_norm)
            torch.nn.utils.clip_grad_norm_(patchmatch_params, max_norm=grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline_params, max_norm=grad_clip_max_norm)
            torch.nn.utils.clip_grad_norm_(patchmatch_params, max_norm=grad_clip_max_norm)
            optimizer.step()

        for key in totals:
            totals[key] += float(loss_terms[key].item())

    if device.type == "cuda":
        torch.cuda.empty_cache()

    steps = max(1, len(train_loader))
    return {key: value / steps for key, value in totals.items()}
