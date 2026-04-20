from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.baseline_config import BaselineConfig, seed_worker, set_seed
from dataset_utils import list_labeled_samples
from datasets.forgery_dataset import ForgeryDataset
from datasets.splits import count_samples_by_split_and_label, make_grouped_stratified_splits
from engine.checkpointing import (
    build_checkpoint_payload,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
    validate_checkpoint_cadence,
)
from engine.train_loop import train_one_epoch
from engine.validate_loop import validate_one_epoch
from inference.sliding_window_dino import sliding_window_dino
from models.dino_segmenter import DinoSegmenter
from util.pixelmapUtil import PixelMapUtil


def main():
    config = BaselineConfig()
    validate_checkpoint_cadence(config.save_last_every_epochs)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using seed: {config.seed}")

    pixel_util = PixelMapUtil()

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
    test_samples = splits["test"]  # keep reference, do not use for training or validation
    split_counts = count_samples_by_split_and_label(splits)
    print(split_counts)

    print(f"Split sizes: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)} (held out)")

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

    train_loader = DataLoader(
        ForgeryDataset(
            train_samples,
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
            val_samples,
            config.target_size,
            use_rgb=config.use_rgb,
            normalize_rgb=config.normalize_rgb,
            rgb_mean=config.dino_mean,
            rgb_std=config.dino_std,
        ),
        batch_size=config.val_batch_size or config.batch_size,
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

    start_epoch = 0
    best_kaggle_score = 0.0
    if config.resume_checkpoint_path:
        # Resume checkpoints are trusted local files configured by the user.
        checkpoint = load_checkpoint(config.resume_checkpoint_path, map_location=device, trusted=True)
        start_epoch, best_kaggle_score = restore_training_state(
            checkpoint=checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            restore_rng=True,
            torch_generators={
                "train_loader": train_loader_generator,
                "val_loader": val_loader_generator,
            },
        )
        print(
            f"Resumed from {config.resume_checkpoint_path} "
            f"at epoch={start_epoch} best_kaggle_score={best_kaggle_score:.4f}"
        )

    for epoch in range(start_epoch, config.num_epochs):
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
        validation_result = validate_one_epoch(
            model=model,
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
            epoch_idx=epoch,
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
            post_process_keep_confident_seeded_components=(
                config.post_process_keep_confident_seeded_components
            ),
        )
        kaggle_score = validation_result["kaggle_score"]
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}  kaggle_score={kaggle_score:.4f}")

        is_new_best = kaggle_score > best_kaggle_score
        if is_new_best:
            best_kaggle_score = kaggle_score

        scheduler.step(kaggle_score)

        if is_new_best:
            best_payload = build_checkpoint_payload(
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                kaggle_score=kaggle_score,
                best_kaggle_score=best_kaggle_score,
                validation_result=validation_result,
                config=config,
                split_counts=split_counts,
                model_name=config.dino_model_name,
                torch_generators={
                    "train_loader": train_loader_generator,
                    "val_loader": val_loader_generator,
                },
            )
            save_checkpoint(best_payload, config.checkpoint_dir, config.best_checkpoint_name)
            print(f"  -> New best model saved by kaggle_score={kaggle_score:.4f}")

        if config.save_last_checkpoint and ((epoch + 1) % config.save_last_every_epochs == 0):
            last_payload = build_checkpoint_payload(
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                kaggle_score=kaggle_score,
                best_kaggle_score=best_kaggle_score,
                validation_result=validation_result,
                config=config,
                split_counts=split_counts,
                model_name=config.dino_model_name,
                torch_generators={
                    "train_loader": train_loader_generator,
                    "val_loader": val_loader_generator,
                },
            )
            save_checkpoint(last_payload, config.checkpoint_dir, config.last_checkpoint_name)


if __name__ == "__main__":
    main()
