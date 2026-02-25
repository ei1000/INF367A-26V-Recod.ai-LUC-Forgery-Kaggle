from pathlib import Path
from functools import partial
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.baseline_config import BaselineConfig, seed_worker, set_seed
from datasets.forgery_dataset import ForgeryDataset
from engine.train_loop import train_one_epoch
from engine.validate_loop import validate_one_epoch
from inference.sliding_window_dino import sliding_window_dino
from models.dino_segmenter import DinoSegmenter
from util.pixelmapUtil import PixelMapUtil


def get_forged_case_ids():
    forged_dir = Path("data/train_images/forged")
    return sorted([p.stem for p in forged_dir.glob("*.png")])


def split_ids(ids, val_ratio=0.1, seed=42):
    rng = random.Random(seed)
    ids = ids.copy()
    rng.shuffle(ids)
    n_val = int(len(ids) * val_ratio)
    return ids[n_val:], ids[:n_val]


def main():
    config = BaselineConfig()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using seed: {config.seed}")

    pixel_util = PixelMapUtil()
    all_ids = get_forged_case_ids()
    train_ids, val_ids = split_ids(all_ids, val_ratio=0.1, seed=config.seed)

    train_ids = train_ids[: config.train_subset]
    val_ids = val_ids[: config.val_subset]

    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(config.seed)
    val_loader_generator = torch.Generator()
    val_loader_generator.manual_seed(config.seed + 1)

    train_loader = DataLoader(
        ForgeryDataset(
            train_ids,
            config.target_size,
            use_rgb=config.use_rgb,
            normalize_rgb=config.normalize_rgb,
            rgb_mean=config.dino_mean,
            rgb_std=config.dino_std,
        ),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.train_num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=train_loader_generator,
    )
    val_loader = DataLoader(
        ForgeryDataset(
            val_ids,
            config.target_size,
            use_rgb=config.use_rgb,
            normalize_rgb=config.normalize_rgb,
            rgb_mean=config.dino_mean,
            rgb_std=config.dino_std,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.val_num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=val_loader_generator,
    )

    model = DinoSegmenter.from_official(
        model_name=config.dino_model_name,
        embed_dim=config.dino_embed_dim,
        freeze_encoder=config.freeze_dino_encoder,
    ).to(device)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.lr,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = bool(config.use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    sw_patch = config.sliding_window_size or config.target_size
    sw_stride = config.sliding_stride or (sw_patch // 2)
    sliding_window_fn = partial(
        sliding_window_dino,
        patch_size=sw_patch,
        stride=sw_stride,
        batch_size=config.sliding_batch_size,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        threshold=1e-4,
    )

    best_f1 = 0.0
    best_model_state = None

    print("Train size:", len(train_ids), "Val size:", len(val_ids))

    for epoch in range(config.num_epochs):
        avg_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip_max_norm=config.grad_clip_max_norm,
            epoch_idx=epoch,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_f1 = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            device=device,
            sliding_window_fn=sliding_window_fn,
            pixel_util=pixel_util,
            pred_threshold=config.pred_threshold,
            harden_temperature=config.harden_temperature,
            hard_clip_low=config.hard_clip_low,
            hard_clip_high=config.hard_clip_high,
            min_component_area=config.min_component_area,
            epoch_idx=epoch,
        )
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()

        scheduler.step(val_f1)


if __name__ == "__main__":
    main()
