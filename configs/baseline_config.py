import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class BaselineConfig:
    num_epochs: int = 10
    batch_size: int = 4
    seed: int = 42
    target_size: int = 448 # Try to keep this divisible by 14. 256 works too, but is small. 
    pred_threshold: float = 0.5
    harden_temperature: float = 0.7
    hard_clip_low: float = 0.1
    hard_clip_high: float = 0.9
    min_component_area: int = 50
    train_subset: int = 200
    val_subset: int = 50
    lr: float = 1e-4
    grad_clip_max_norm: float = 1.0
    train_num_workers: int = 2
    val_num_workers: int = 1
    use_rgb: bool = True
    normalize_rgb: bool = True
    dino_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    dino_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    dino_model_name: str = "dinov2_vitb14"
    dino_embed_dim: int = 768
    freeze_dino_encoder: bool = True
    use_amp: bool = True
    sliding_window_size: int | None = 448
    sliding_stride: int | None = 224
    sliding_batch_size: int = 8


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
