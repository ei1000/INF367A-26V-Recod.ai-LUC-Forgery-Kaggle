from __future__ import annotations

from pathlib import Path

import torch


def ensure_output_dirs(output_dir: str | Path):
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    predictions_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, checkpoints_dir, predictions_dir


def save_checkpoint(path: Path, epoch: int, dlf_decoder, se_model, optimizer, best_score: float | None, pyramid_bb=None):
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
