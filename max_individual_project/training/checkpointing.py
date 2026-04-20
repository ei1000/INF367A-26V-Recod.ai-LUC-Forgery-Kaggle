from __future__ import annotations

from pathlib import Path

import torch

try:
    from .optim import set_optimizer_learning_rate
except ImportError:
    from training.optim import set_optimizer_learning_rate


def ensure_output_dirs(output_dir: str | Path):
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    predictions_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, checkpoints_dir, predictions_dir


def load_resume_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    resume: bool = True,
) -> tuple[dict | None, int, float | None]:
    if not resume or not checkpoint_path.exists():
        return None, 0, None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    resume_epoch = int(checkpoint.get("epoch", 0))
    best_score = checkpoint.get("best_score", checkpoint.get("best_loss"))
    print(f"[pipeline] Resuming from checkpoint: {checkpoint_path}")
    return checkpoint, resume_epoch, best_score


def save_checkpoint(
    path: Path,
    epoch: int,
    dlf_decoder,
    se_model,
    optimizer,
    best_score: float | None,
    pyramid_bb=None,
    dino_extractor=None,
    pm_backbone=None,
):
    checkpoint = {
        "epoch": epoch,
        "dlf_decoder": dlf_decoder.state_dict(),
        "se_model": se_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "best_score": best_score,
        "best_loss": best_score,
    }
    if pyramid_bb is not None:
        checkpoint["pyramid_bb"] = pyramid_bb.state_dict()
    if dino_extractor is not None:
        checkpoint["dino_extractor"] = dino_extractor.state_dict()
    if pm_backbone is not None:
        checkpoint["pm_backbone"] = pm_backbone.state_dict()
    torch.save(checkpoint, path)


def save_prediction_batch(predictions_dir: Path, epoch_idx: int, batch_idx: int, mask_preds: torch.Tensor):
    pred_path = predictions_dir / f"epoch_{epoch_idx + 1:03d}_batch_{batch_idx:05d}.pt"
    torch.save(mask_preds.detach().cpu(), pred_path)


def load_module_state(module, state_dict: dict[str, torch.Tensor], module_name: str) -> bool:
    try:
        incompatible = module.load_state_dict(state_dict, strict=False)
    except RuntimeError:
        current_state = module.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        missing_keys = sorted(set(current_state.keys()) - set(compatible_state.keys()))
        skipped_keys = sorted(set(state_dict.keys()) - set(compatible_state.keys()))
        module.load_state_dict(compatible_state, strict=False)
        print(
            f"[pipeline] Partially restored {module_name}: "
            f"loaded {len(compatible_state)} tensors, skipped {len(skipped_keys)}, missing {len(missing_keys)}."
        )
        return False

    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        print(
            f"[pipeline] Restored {module_name} with non-strict loading: "
            f"missing {len(missing)} keys, unexpected {len(unexpected)}."
        )
        return False
    return True


def restore_training_state(
    checkpoint: dict | None,
    pm_backbone,
    dino_extractor,
    dlf_decoder,
    se_model,
    optimizer,
    learning_rate: float,
) -> dict | None:
    if checkpoint is None:
        return None

    fully_restored = True
    if "pm_backbone" in checkpoint:
        fully_restored = load_module_state(pm_backbone, checkpoint["pm_backbone"], "pm_backbone") and fully_restored
    if "dino_extractor" in checkpoint:
        fully_restored = load_module_state(dino_extractor, checkpoint["dino_extractor"], "dino_extractor") and fully_restored
    elif "pyramid_bb" in checkpoint:
        fully_restored = load_module_state(dino_extractor, checkpoint["pyramid_bb"], "dino_extractor") and fully_restored
    if checkpoint.get("dlf_decoder") is not None:
        fully_restored = load_module_state(dlf_decoder, checkpoint["dlf_decoder"], "dlf_decoder") and fully_restored
    if checkpoint.get("se_model") is not None:
        fully_restored = load_module_state(se_model, checkpoint["se_model"], "se_model") and fully_restored
    if optimizer is not None and checkpoint.get("optimizer") is not None and fully_restored:
        optimizer.load_state_dict(checkpoint["optimizer"])
        set_optimizer_learning_rate(optimizer, learning_rate)
    elif optimizer is not None and checkpoint.get("optimizer") is not None:
        print("[pipeline] Skipping optimizer restore because model weights were only partially restored.")

    return None


def save_epoch_checkpoints(
    checkpoint_path: Path,
    best_checkpoint_path: Path,
    epoch: int,
    dlf_decoder,
    se_model,
    optimizer,
    checkpoint_score: float,
    best_score: float | None,
    dino_extractor=None,
    pm_backbone=None,
) -> float:
    save_checkpoint(
        checkpoint_path,
        epoch=epoch,
        dlf_decoder=dlf_decoder,
        se_model=se_model,
        optimizer=optimizer,
        best_score=checkpoint_score,
        dino_extractor=dino_extractor,
        pm_backbone=pm_backbone,
    )
    if best_score is None or checkpoint_score > best_score:
        best_score = checkpoint_score
        save_checkpoint(
            best_checkpoint_path,
            epoch=epoch,
            dlf_decoder=dlf_decoder,
            se_model=se_model,
            optimizer=optimizer,
            best_score=best_score,
            dino_extractor=dino_extractor,
            pm_backbone=pm_backbone,
        )
    return best_score
