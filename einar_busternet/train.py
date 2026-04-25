from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_utils import list_labeled_samples
from datasets.forgery_dataset import ForgeryDataset
from datasets.splits import count_samples_by_split_and_label, make_grouped_stratified_splits
from einar_busternet.config import BusterNetConfig, seed_worker, set_seed
from einar_busternet.dataset import BusterNetDataset
from einar_busternet.model import BusterNetUnionWrapper, DinoBusterNet
from engine.checkpointing import (
    build_checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_cadence,
)
from engine.train_loop import train_one_epoch
from engine.validate_loop import validate_one_epoch
from inference.sliding_window_dino import sliding_window_dino
from util.pixelmapUtil import PixelMapUtil


def build_config_from_args(argv: Sequence[str] | None = None) -> BusterNetConfig:
    parser = argparse.ArgumentParser(description="Train DINO-BusterNet.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny end-to-end sanity check.")
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--val-subset", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--stage2-epochs", type=int, default=None)
    parser.add_argument("--stage3-epochs", type=int, default=None)
    parser.add_argument("--resume-checkpoint-path", type=str, default=None)
    parser.add_argument(
        "--validation-transfer-mode",
        choices=("per_batch", "accumulate_gpu"),
        default=None,
    )
    args = parser.parse_args(argv)

    config = BusterNetConfig()
    overrides = {}
    for arg_name, field_name in (
        ("train_subset", "train_subset"),
        ("val_subset", "val_subset"),
        ("batch_size", "batch_size"),
        ("stage1_epochs", "stage1_epochs"),
        ("stage2_epochs", "stage2_epochs"),
        ("stage3_epochs", "stage3_epochs"),
        ("resume_checkpoint_path", "resume_checkpoint_path"),
        ("validation_transfer_mode", "validation_transfer_mode"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = value

    if args.smoke:
        overrides.update(
            {
                "train_subset": min(overrides.get("train_subset", 8), 8),
                "val_subset": min(overrides.get("val_subset", 8), 8),
                "batch_size": min(overrides.get("batch_size", 2), 2),
                "stage1_epochs": min(overrides.get("stage1_epochs", 1), 1),
                "stage2_epochs": min(overrides.get("stage2_epochs", 1), 1),
                "stage3_epochs": min(overrides.get("stage3_epochs", 0), 0),
                "validation_log_timing": False,
            }
        )

    return replace(config, **overrides)


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = enabled


def configure_trainable_parts(model: DinoBusterNet, stage: int) -> None:
    if stage == 1:
        _set_requires_grad(model.mani_decoder, True)
        _set_requires_grad(model.simi_decoder, True)
        _set_requires_grad(model.fusion, False)
    elif stage == 2:
        _set_requires_grad(model.mani_decoder, False)
        _set_requires_grad(model.simi_decoder, False)
        _set_requires_grad(model.fusion, True)
    elif stage == 3:
        _set_requires_grad(model.mani_decoder, True)
        _set_requires_grad(model.simi_decoder, True)
        _set_requires_grad(model.fusion, True)
    else:
        raise ValueError(f"stage must be 1, 2, or 3, got {stage}")

    model.freeze_encoder()


def _trainable_params(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
    return [param for module in modules for param in module.parameters() if param.requires_grad]


def train_stage1_epoch(
    *,
    model: DinoBusterNet,
    train_loader,
    mani_optimizer: torch.optim.Optimizer,
    simi_optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip_max_norm: float,
    epoch_idx: int,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = torch.zeros((), device=device)
    total_mani_loss = torch.zeros((), device=device)
    total_simi_loss = torch.zeros((), device=device)

    branch_params = _trainable_params((model.mani_decoder, model.simi_decoder))
    progress = tqdm(train_loader, desc=f"stage 1 epoch {epoch_idx + 1} train")
    for imgs, labels in progress:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        target_mask = (labels == 1).float()
        source_mask = (labels == 2).float()

        mani_optimizer.zero_grad(set_to_none=True)
        simi_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            mani_logits, simi_logits = model.forward_branches(imgs)
            mani_loss = loss_fn(mani_logits[:, 1], target_mask)
            simi_loss = loss_fn(simi_logits[:, 2], source_mask)
            loss = mani_loss + simi_loss

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(mani_optimizer)
            scaler.unscale_(simi_optimizer)
            torch.nn.utils.clip_grad_norm_(branch_params, max_norm=grad_clip_max_norm)
            scaler.step(mani_optimizer)
            scaler.step(simi_optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(branch_params, max_norm=grad_clip_max_norm)
            mani_optimizer.step()
            simi_optimizer.step()

        total_loss += loss.detach()
        total_mani_loss += mani_loss.detach()
        total_simi_loss += simi_loss.detach()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    denom = max(1, len(train_loader))
    return {
        "loss": float((total_loss / denom).item()),
        "mani_loss": float((total_mani_loss / denom).item()),
        "simi_loss": float((total_simi_loss / denom).item()),
    }


def _build_loaders(config: BusterNetConfig, device: torch.device):
    all_samples = list_labeled_samples(Path(config.data_root))
    splits = make_grouped_stratified_splits(
        all_samples,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    train_samples = splits["train"]
    val_samples = splits["val"]
    test_samples = splits["test"]
    split_counts = count_samples_by_split_and_label(splits)

    if config.train_subset is not None:
        print(f"Debug train_subset enabled: using {config.train_subset} of {len(train_samples)} train samples")
        train_samples = train_samples[: config.train_subset]
    if config.val_subset is not None:
        print(f"Debug val_subset enabled: using {config.val_subset} of {len(val_samples)} val samples")
        val_samples = val_samples[: config.val_subset]

    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(config.seed)
    val_loader_generator = torch.Generator()
    val_loader_generator.manual_seed(config.seed + 1)

    train_dataset = BusterNetDataset(
        train_samples,
        data_root=config.data_root,
        target_size=config.target_size,
        use_rgb=config.use_rgb,
        normalize_rgb=config.normalize_rgb,
        rgb_mean=config.dino_mean,
        rgb_std=config.dino_std,
        metadata_path=config.metadata_path,
        allowed_forged_statuses=config.allowed_forged_statuses,
        include_authentic=config.include_authentic,
        authentic_policy=config.authentic_policy,
    )
    val_dataset = ForgeryDataset(
        val_samples,
        config.target_size,
        use_rgb=config.use_rgb,
        normalize_rgb=config.normalize_rgb,
        rgb_mean=config.dino_mean,
        rgb_std=config.dino_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.train_num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=train_loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size or config.batch_size,
        shuffle=False,
        num_workers=config.val_num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=val_loader_generator,
    )
    return (
        train_loader,
        val_loader,
        train_dataset.samples,
        val_samples,
        test_samples,
        split_counts,
        train_loader_generator,
        val_loader_generator,
    )


def _build_sliding_window_fn(config: BusterNetConfig):
    sw_patch = config.sliding_window_size or config.target_size
    sw_stride = config.sliding_stride or (sw_patch // 2)
    return partial(
        sliding_window_dino,
        patch_size=sw_patch,
        stride=sw_stride,
        batch_size=config.sliding_batch_size,
    )


def _validate_model(
    *,
    model: DinoBusterNet,
    config: BusterNetConfig,
    val_loader,
    val_samples,
    device: torch.device,
    sliding_window_fn,
    pixel_util,
    epoch_idx: int,
) -> dict:
    wrapped_model = BusterNetUnionWrapper(model, eps=config.union_wrapper_eps)
    return validate_one_epoch(
        model=wrapped_model,
        val_loader=val_loader,
        val_samples=val_samples,
        device=device,
        sliding_window_fn=sliding_window_fn,
        pixel_util=pixel_util,
        pred_threshold=config.pred_threshold,
        harden_temperature=config.harden_temperature,
        hard_clip_low=config.hard_clip_low,
        hard_clip_high=config.hard_clip_high,
        min_component_area=config.min_component_area,
        epoch_idx=epoch_idx,
        compute_pixel_f1=config.compute_pixel_f1,
        verify_score_equivalence=config.verify_score_equivalence,
        inference_mode=config.validation_inference_mode,
        probability_dtype=config.validation_probability_dtype,
        validation_transfer_mode=config.validation_transfer_mode,
        log_timing=config.validation_log_timing,
        post_process_confident_threshold=config.post_process_confident_threshold,
        post_process_smooth_probabilities=config.post_process_smooth_probabilities,
        post_process_fill_holes=config.post_process_fill_holes,
        post_process_apply_opening=config.post_process_apply_opening,
        post_process_apply_closing=config.post_process_apply_closing,
        post_process_keep_confident_seeded_components=config.post_process_keep_confident_seeded_components,
    )


def _save_training_checkpoint(
    *,
    model: DinoBusterNet,
    optimizer: torch.optim.Optimizer | None,
    scheduler,
    scaler,
    config: BusterNetConfig,
    split_counts: dict,
    epoch: int,
    stage: int,
    train_loss: float,
    validation_result: dict,
    kaggle_score: float,
    best_kaggle_score: float,
    checkpoint_name: str,
    train_loader_generator: torch.Generator,
    val_loader_generator: torch.Generator,
) -> None:
    payload = build_checkpoint_payload(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        kaggle_score=kaggle_score,
        best_kaggle_score=best_kaggle_score,
        validation_result={
            **validation_result,
            "stage": stage,
            "train_loss": float(train_loss),
        },
        config=config,
        split_counts=split_counts,
        model_name=config.dino_model_name,
        torch_generators={
            "train_loader": train_loader_generator,
            "val_loader": val_loader_generator,
        },
    )
    save_checkpoint(payload, config.checkpoint_dir, checkpoint_name)


def main(config: BusterNetConfig | None = None) -> None:
    if config is None:
        config = build_config_from_args()
    validate_checkpoint_cadence(config.save_last_every_epochs)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using seed: {config.seed}")

    pixel_util = PixelMapUtil()
    (
        train_loader,
        val_loader,
        effective_train_samples,
        val_samples,
        test_samples,
        split_counts,
        train_loader_generator,
        val_loader_generator,
    ) = _build_loaders(config, device)
    print(split_counts)
    print(
        "Split sizes: "
        f"train_effective={len(effective_train_samples)}, "
        f"val={len(val_samples)}, test={len(test_samples)} (held out)"
    )

    model = DinoBusterNet.from_official(
        model_name=config.dino_model_name,
        embed_dim=config.dino_embed_dim,
        nb_pools=config.nb_pools,
        freeze_encoder=config.freeze_dino_encoder,
    ).to(device)
    if config.resume_checkpoint_path:
        checkpoint = load_checkpoint(config.resume_checkpoint_path, map_location=device, trusted=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model weights from {config.resume_checkpoint_path}")

    use_amp = bool(config.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    sliding_window_fn = _build_sliding_window_fn(config)
    best_kaggle_score = 0.0
    global_epoch = 0

    configure_trainable_parts(model, stage=1)
    mani_optimizer = torch.optim.Adam(_trainable_params((model.mani_decoder,)), lr=config.stage1_lr)
    simi_optimizer = torch.optim.Adam(_trainable_params((model.simi_decoder,)), lr=config.stage1_lr)
    bce_loss = nn.BCEWithLogitsLoss()
    for epoch in range(config.stage1_epochs):
        metrics = train_stage1_epoch(
            model=model,
            train_loader=train_loader,
            mani_optimizer=mani_optimizer,
            simi_optimizer=simi_optimizer,
            loss_fn=bce_loss,
            device=device,
            grad_clip_max_norm=config.grad_clip_max_norm,
            epoch_idx=epoch,
            use_amp=use_amp,
            scaler=scaler,
        )
        global_epoch += 1
        print(
            f"Stage 1 epoch {epoch + 1}: "
            f"loss={metrics['loss']:.4f} mani={metrics['mani_loss']:.4f} simi={metrics['simi_loss']:.4f}"
        )
        if config.save_last_checkpoint and (global_epoch % config.save_last_every_epochs == 0):
            _save_training_checkpoint(
                model=model,
                optimizer=mani_optimizer,
                scheduler=None,
                scaler=scaler,
                config=config,
                split_counts=split_counts,
                epoch=global_epoch,
                stage=1,
                train_loss=metrics["loss"],
                validation_result={"kaggle_score": 0.0},
                kaggle_score=0.0,
                best_kaggle_score=best_kaggle_score,
                checkpoint_name=config.last_checkpoint_name,
                train_loader_generator=train_loader_generator,
                val_loader_generator=val_loader_generator,
            )

    ce_weights = torch.tensor(config.ce_class_weights, dtype=torch.float32, device=device)
    ce_loss = nn.CrossEntropyLoss(weight=ce_weights)

    configure_trainable_parts(model, stage=2)
    fusion_optimizer = torch.optim.Adam(_trainable_params((model.fusion,)), lr=config.stage2_lr)
    for epoch in range(config.stage2_epochs):
        avg_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=fusion_optimizer,
            loss_fn=ce_loss,
            device=device,
            grad_clip_max_norm=config.grad_clip_max_norm,
            epoch_idx=epoch,
            use_amp=use_amp,
            scaler=scaler,
        )
        validation_result = _validate_model(
            model=model,
            config=config,
            val_loader=val_loader,
            val_samples=val_samples,
            device=device,
            sliding_window_fn=sliding_window_fn,
            pixel_util=pixel_util,
            epoch_idx=global_epoch,
        )
        global_epoch += 1
        kaggle_score = validation_result["kaggle_score"]
        print(f"Stage 2 epoch {epoch + 1}: avg_loss={avg_loss:.4f} kaggle_score={kaggle_score:.4f}")
        if kaggle_score > best_kaggle_score:
            best_kaggle_score = kaggle_score
            _save_training_checkpoint(
                model=model,
                optimizer=fusion_optimizer,
                scheduler=None,
                scaler=scaler,
                config=config,
                split_counts=split_counts,
                epoch=global_epoch,
                stage=2,
                train_loss=avg_loss,
                validation_result=validation_result,
                kaggle_score=kaggle_score,
                best_kaggle_score=best_kaggle_score,
                checkpoint_name=config.best_checkpoint_name,
                train_loader_generator=train_loader_generator,
                val_loader_generator=val_loader_generator,
            )
            print(f"  -> New best BusterNet saved by kaggle_score={kaggle_score:.4f}")
        if config.save_last_checkpoint and (global_epoch % config.save_last_every_epochs == 0):
            _save_training_checkpoint(
                model=model,
                optimizer=fusion_optimizer,
                scheduler=None,
                scaler=scaler,
                config=config,
                split_counts=split_counts,
                epoch=global_epoch,
                stage=2,
                train_loss=avg_loss,
                validation_result=validation_result,
                kaggle_score=kaggle_score,
                best_kaggle_score=best_kaggle_score,
                checkpoint_name=config.last_checkpoint_name,
                train_loader_generator=train_loader_generator,
                val_loader_generator=val_loader_generator,
            )

    configure_trainable_parts(model, stage=3)
    stage3_optimizer = torch.optim.Adam(
        _trainable_params((model.mani_decoder, model.simi_decoder, model.fusion)),
        lr=config.stage3_lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        stage3_optimizer,
        mode="max",
        factor=config.stage3_scheduler_factor,
        patience=config.stage3_scheduler_patience,
        threshold=1e-4,
    )
    epochs_without_improvement = 0
    for epoch in range(config.stage3_epochs):
        avg_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=stage3_optimizer,
            loss_fn=ce_loss,
            device=device,
            grad_clip_max_norm=config.grad_clip_max_norm,
            epoch_idx=epoch,
            use_amp=use_amp,
            scaler=scaler,
        )
        validation_result = _validate_model(
            model=model,
            config=config,
            val_loader=val_loader,
            val_samples=val_samples,
            device=device,
            sliding_window_fn=sliding_window_fn,
            pixel_util=pixel_util,
            epoch_idx=global_epoch,
        )
        global_epoch += 1
        kaggle_score = validation_result["kaggle_score"]
        print(f"Stage 3 epoch {epoch + 1}: avg_loss={avg_loss:.4f} kaggle_score={kaggle_score:.4f}")

        is_new_best = kaggle_score > best_kaggle_score
        if is_new_best:
            best_kaggle_score = kaggle_score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        scheduler.step(kaggle_score)

        if is_new_best:
            _save_training_checkpoint(
                model=model,
                optimizer=stage3_optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                split_counts=split_counts,
                epoch=global_epoch,
                stage=3,
                train_loss=avg_loss,
                validation_result=validation_result,
                kaggle_score=kaggle_score,
                best_kaggle_score=best_kaggle_score,
                checkpoint_name=config.best_checkpoint_name,
                train_loader_generator=train_loader_generator,
                val_loader_generator=val_loader_generator,
            )
            print(f"  -> New best BusterNet saved by kaggle_score={kaggle_score:.4f}")
        if config.save_last_checkpoint and (global_epoch % config.save_last_every_epochs == 0):
            _save_training_checkpoint(
                model=model,
                optimizer=stage3_optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                split_counts=split_counts,
                epoch=global_epoch,
                stage=3,
                train_loss=avg_loss,
                validation_result=validation_result,
                kaggle_score=kaggle_score,
                best_kaggle_score=best_kaggle_score,
                checkpoint_name=config.last_checkpoint_name,
                train_loader_generator=train_loader_generator,
                val_loader_generator=val_loader_generator,
            )
        if epochs_without_improvement >= config.early_stop_patience:
            print(f"Stage 3 early stop after {epochs_without_improvement} epochs without improvement")
            break


if __name__ == "__main__":
    main()
