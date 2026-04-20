from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import random

import numpy as np
import torch


def validate_checkpoint_cadence(save_last_every_epochs: int) -> None:
    if save_last_every_epochs < 1:
        raise ValueError("save_last_every_epochs must be at least 1")


def capture_rng_state(
    torch_generators: Mapping[str, torch.Generator] | None = None,
) -> dict[str, Any]:
    rng_state: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if torch_generators:
        rng_state["torch_generators"] = {
            name: generator.get_state() for name, generator in torch_generators.items()
        }
    return rng_state


def restore_rng_state(
    rng_state: dict[str, Any] | None,
    torch_generators: Mapping[str, torch.Generator] | None = None,
) -> None:
    if not rng_state:
        return

    python_state = rng_state.get("python_random_state")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = rng_state.get("numpy_random_state")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_state = rng_state.get("torch_rng_state")
    if torch_state is not None:
        torch.set_rng_state(torch_state)

    cuda_state = rng_state.get("cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)

    generator_states = rng_state.get("torch_generators")
    if generator_states and torch_generators:
        for name, generator in torch_generators.items():
            state = generator_states.get(name)
            if state is not None:
                generator.set_state(state)


def _state_dict_or_none(obj: Any) -> Any:
    if obj is None or not hasattr(obj, "state_dict"):
        return None
    return obj.state_dict()


def build_checkpoint_payload(
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    kaggle_score: float,
    best_kaggle_score: float,
    validation_result: dict,
    config,
    split_counts: dict,
    model_name: str,
    torch_generators: Mapping[str, torch.Generator] | None = None,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": _state_dict_or_none(optimizer),
        "scheduler_state_dict": _state_dict_or_none(scheduler),
        "scaler_state_dict": _state_dict_or_none(scaler),
        "kaggle_score": float(kaggle_score),
        "best_kaggle_score": float(best_kaggle_score),
        "validation_result": validation_result,
        "config": dict(getattr(config, "__dict__", config)),
        "split_counts": split_counts,
        "model_name": model_name,
        "rng_state": capture_rng_state(torch_generators=torch_generators),
    }


def save_checkpoint(payload: dict[str, Any], checkpoint_dir: str | Path, checkpoint_name: str) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / checkpoint_name
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(path: str | Path, map_location="cpu", *, trusted: bool = False) -> dict[str, Any]:
    if not trusted:
        raise ValueError(
            "Full checkpoint loading uses pickle and requires trusted=True for trusted local checkpoints."
        )
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def restore_training_state(
    *,
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler=None,
    restore_rng: bool = True,
    torch_generators: Mapping[str, torch.Generator] | None = None,
) -> tuple[int, float]:
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    if restore_rng:
        restore_rng_state(checkpoint.get("rng_state"), torch_generators=torch_generators)

    start_epoch = int(checkpoint.get("epoch", 0))
    best_kaggle_score = float(checkpoint.get("best_kaggle_score", checkpoint.get("kaggle_score", 0.0)))
    return start_epoch, best_kaggle_score
