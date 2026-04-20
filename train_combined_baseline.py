from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.baseline_config import seed_worker, set_seed
from configs.combined_baseline_config import CombinedBaselineConfig
from dataset_utils import list_labeled_samples
from datasets.forgery_dataset import ForgeryDataset
from datasets.splits import count_samples_by_split_and_label, make_grouped_stratified_splits
from engine.combined_train_loop import (
    build_combined_optimizer,
    restore_optimizer_state,
    train_combined_one_epoch,
)
from engine.validate_loop import validate_one_epoch
from models.combined_baseline import CombinedBaselineModel
from util.pixelmapUtil import PixelMapUtil


def _load_resume_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    resume: bool,
) -> tuple[dict | None, int, float]:
    if not resume or not checkpoint_path.exists():
        return None, 0, 0.0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    resume_epoch = int(checkpoint.get("epoch", 0))
    best_score = float(checkpoint.get("best_score", 0.0))
    print(f"Resuming combined pipeline from checkpoint: {checkpoint_path}")
    return checkpoint, resume_epoch, best_score


def _select_initialization_batch(train_loader, val_loader) -> torch.Tensor:
    for loader in (train_loader, val_loader):
        dataset = getattr(loader, "dataset", None)
        if dataset is not None and len(dataset) > 0:
            sample = dataset[0]
            image = sample[0]
            if image.dim() != 3:
                raise ValueError(f"Expected dataset image shape [C,H,W], got {tuple(image.shape)}")
            return image.unsqueeze(0)
    raise ValueError("Could not initialize PatchMatch decoder because both train and val datasets are empty.")


def _save_combined_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: CombinedBaselineModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    best_score: float,
    validation_result: dict,
    config: CombinedBaselineConfig,
    split_counts: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "baseline_model_state_dict": model.baseline_model.state_dict(),
            "patchmatch_decoder_state_dict": model.patchmatch_decoder_state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_score": best_score,
            "validation_result": validation_result,
            "config": config.__dict__,
            "split_counts": split_counts,
        },
        path,
    )


def main(config: CombinedBaselineConfig | None = None) -> None:
    config = config or CombinedBaselineConfig()
    if config.validation_inference_mode != "direct":
        raise ValueError(
            "Combined baseline validation currently supports direct inference only, "
            f"got {config.validation_inference_mode!r}."
        )

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using seed: {config.seed}")

    run_dir = Path(config.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    predictions_dir = run_dir / "predictions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    latest_checkpoint_path = checkpoint_dir / config.latest_checkpoint_name
    best_checkpoint_path = checkpoint_dir / config.best_checkpoint_name
    checkpoint, resume_epoch, best_kaggle_score = _load_resume_checkpoint(
        latest_checkpoint_path,
        device=device,
        resume=config.resume,
    )

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
    test_samples = splits["test"]
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
            use_rgb=True,
            normalize_rgb=False,
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
            use_rgb=True,
            normalize_rgb=False,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.val_num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=val_loader_generator,
    )

    model = CombinedBaselineModel(
        device=device,
        dino_model_name=config.dino_model_name,
        dino_embed_dim=config.dino_embed_dim,
        freeze_dino_encoder=config.freeze_dino_encoder,
        dino_mean=config.dino_mean,
        dino_std=config.dino_std,
        separate_transforms=config.separate_transforms,
        cnn_feature_norm=config.cnn_feature_norm,
        pm_random_window=config.pm_random_window,
        pm_iters=config.pm_iters,
        pm_beta=config.pm_beta,
        pm_hard_selection=config.pm_hard_selection,
        pm_use_non_local=config.pm_use_non_local,
        pm_non_local_limit=config.pm_non_local_limit,
        pm_flat_threshold=config.pm_flat_threshold,
        pm_margin_threshold=config.pm_margin_threshold,
        pm_topk=config.pm_topk,
        pm_reduced_precision=config.pm_reduced_precision,
        dlf_error_scaling=config.dlf_error_scaling,
        patchmatch_pre_fusion_postprocess=config.patchmatch_pre_fusion_postprocess,
        patchmatch_pre_fusion_threshold=config.patchmatch_pre_fusion_threshold,
        patchmatch_pre_fusion_confident_threshold=config.patchmatch_pre_fusion_confident_threshold,
        patchmatch_pre_fusion_min_component_area=config.patchmatch_pre_fusion_min_component_area,
        patchmatch_pre_fusion_smooth_probabilities=config.patchmatch_pre_fusion_smooth_probabilities,
        patchmatch_pre_fusion_fill_holes=config.patchmatch_pre_fusion_fill_holes,
        patchmatch_pre_fusion_apply_closing=config.patchmatch_pre_fusion_apply_closing,
    )
    if checkpoint is not None:
        model.baseline_model.load_state_dict(checkpoint["baseline_model_state_dict"])
        model.set_pending_patchmatch_decoder_state(checkpoint.get("patchmatch_decoder_state_dict"))

    init_batch = _select_initialization_batch(train_loader, val_loader).to(device)
    model.initialize_patchmatch_decoder(init_batch)

    optimizer = build_combined_optimizer(model, config.lr)
    optimizer_state = checkpoint.get("optimizer_state_dict") if checkpoint is not None else None
    if checkpoint is None or model.patchmatch_decoder_restored_fully:
        restore_optimizer_state(optimizer, optimizer_state, config.lr)
    elif optimizer_state is not None:
        print("Skipping optimizer restore because the PatchMatch decoder was only partially restored.")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        threshold=1e-4,
    )
    if checkpoint is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    loss_fn = nn.BCEWithLogitsLoss()
    use_amp = bool(config.use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if (
        checkpoint is not None
        and checkpoint.get("scaler_state_dict") is not None
        and use_amp
    ):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    for epoch in range(resume_epoch, config.num_epochs):
        train_summary = train_combined_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip_max_norm=config.grad_clip_max_norm,
            epoch_idx=epoch,
            patchmatch_loss_weight=config.patchmatch_loss_weight,
            use_amp=use_amp,
            scaler=scaler,
        )
        validation_result = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            val_samples=val_samples,
            device=device,
            sliding_window_fn=None,
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
            log_timing=config.validation_log_timing,
            save_prediction_tensors=config.save_predictions,
            prediction_output_dir=predictions_dir,
        )
        kaggle_score = validation_result["kaggle_score"]
        print(
            f"Epoch {epoch + 1}: "
            f"loss={train_summary['loss']:.4f} "
            f"combined_bce={train_summary['combined_bce']:.4f} "
            f"baseline_dice={train_summary['baseline_dice']:.4f} "
            f"patchmatch_dice={train_summary['patchmatch_dice']:.4f} "
            f"kaggle_score={kaggle_score:.4f}"
        )

        scheduler.step(kaggle_score)
        current_best_score = max(best_kaggle_score, kaggle_score)
        _save_combined_checkpoint(
            latest_checkpoint_path,
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if use_amp else None,
            best_score=current_best_score,
            validation_result=validation_result,
            config=config,
            split_counts=split_counts,
        )

        if kaggle_score > best_kaggle_score:
            best_kaggle_score = kaggle_score
            _save_combined_checkpoint(
                best_checkpoint_path,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if use_amp else None,
                best_score=best_kaggle_score,
                validation_result=validation_result,
                config=config,
                split_counts=split_counts,
            )
            print(f"  -> New best combined model saved by kaggle_score={kaggle_score:.4f}")


if __name__ == "__main__":
    main()
