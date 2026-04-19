from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from dataset import (
    ForgeryDataset,
    combine_datasets,
    resolve_data_root,
    resolve_image_transform,
    split_indices_by_label_three_way,
)
from pipeline_config import PipelineConfig, resolve_pipeline_config
from pipeline_helpers import (
    build_localization_optimizer,
    build_patchmatch_feature_branch,
    build_patchmatch_head,
    build_seunet_feature_branch,
    build_seunet_head,
    post_process_predictions,
    set_frozen_feature_branch_modes,
    set_trainable_head_modes,
)
from prediction.localization import decode_and_refine_masks, extract_localization_inputs
from prediction.mask_metrics import (
    binary_mask_to_instances,
    initialize_segmentation_counts,
    load_resized_gt_instances,
    optimal_f1_score,
    summarize_segmentation_counts,
    update_segmentation_counts,
)
from prediction.pixelmaputil_mask import MaskUtil
from training.checkpointing import (
    ensure_output_dirs,
    load_resume_checkpoint,
    restore_training_state,
    save_epoch_checkpoints,
    save_prediction_batch,
)
from training.losses import localization_loss_terms, summarize_branch_activity
from training.metrics_logging import (
    append_metrics_log,
    average_metric_accumulator,
    build_validation_summary,
    build_epoch_metrics,
    format_split_summary,
    format_train_batch_message,
    format_train_epoch_message,
    format_validation_message,
    initialize_instance_metric_tracker,
    initialize_metric_accumulator,
    save_metrics_plot,
    summarize_metric_step,
    update_instance_metric_tracker,
    update_metric_accumulator,
    write_split_artifacts,
)
from visualizer import display_image, display_pixel_offsets


def pipeline(config: PipelineConfig | None = None, **overrides):
    """Train or inspect the copy-move localization pipeline.

    Current architecture:
    - frozen ResNet18 PatchMatch branch
    - frozen multi-scale Zernike PatchMatch branch
    - frozen DINO features feeding an SEUNet refinement head
    """

    config = resolve_pipeline_config(config, **overrides)
    datasets = config.datasets
    image_size = config.image_size
    epochs = config.epochs
    test_run = config.test_run
    feature_backbone = config.feature_backbone
    use_dino_transform = config.use_dino_transform
    batch_size = config.batch_size
    override_batch_size = config.override_batch_size
    dino_model_name = config.dino_model_name
    cnn_backbone = config.cnn_backbone
    cnn_feature_norm = config.cnn_feature_norm
    separate_transforms = config.separate_transforms
    pm_iters = config.pm_iters
    pm_beta = config.pm_beta
    pm_hard_selection = config.pm_hard_selection
    pm_random_window = config.pm_random_window
    pm_use_non_local = config.pm_use_non_local
    pm_non_local_limit = config.pm_non_local_limit
    pm_flat_threshold = config.pm_flat_threshold
    pm_margin_threshold = config.pm_margin_threshold
    pm_topk = config.pm_topk
    pm_reduced_precision = config.pm_reduced_precision
    localization_resolution = config.localization_resolution
    log_every = config.log_every
    output_dir = config.output_dir
    checkpoint_name = config.checkpoint_name
    resume = config.resume
    save_predictions = config.save_predictions
    validation_split = config.validation_split
    test_split = config.test_split
    validation_seed = config.validation_seed
    learning_rate = config.learning_rate
    mprime_loss_weight = config.mprime_loss_weight
    empty_target_penalty_weight = config.empty_target_penalty_weight
    dlf_error_scaling = config.dlf_error_scaling
    do_post_process = config.do_post_process
    post_process_threshold = config.post_process_threshold
    post_process_confident_threshold = config.post_process_confident_threshold
    post_process_smooth_probabilities = config.post_process_smooth_probabilities
    post_process_fill_holes = config.post_process_fill_holes
    post_process_apply_closing = config.post_process_apply_closing
    post_process_min_component_area = config.post_process_min_component_area

    print('[pipeline] Initializing training loop and datasets...')
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir, checkpoints_dir, predictions_dir = ensure_output_dirs(output_dir)
    checkpoint_path = checkpoints_dir / checkpoint_name
    best_checkpoint_path = checkpoints_dir / "best.pt"
    checkpoint, resume_epoch, best_score = load_resume_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        resume=resume,
    )

    root = resolve_data_root()

    transform = resolve_image_transform(
        feature_backbone=feature_backbone,
        use_dino_transform=use_dino_transform,
        cnn_backbone=cnn_backbone,
        separate_transforms=separate_transforms,
    )

    if batch_size > 8 and not override_batch_size:
        print("[pipeline] PatchMatch is memory-heavy; forcing batch_size=8 for 16GB VRAM safety")
        print("[pipeline] Override on powerful devices by adding override=True")
        batch_size = 8
    if feature_backbone == "dino_single" and image_size > 1024 and batch_size > 1 and not override_batch_size:
        print("[pipeline] Single-scale DINO at large image sizes is memory-heavy; forcing batch_size=1 for 16GB VRAM safety")
        print("[pipeline] Override on powerful devices by adding override=True")
        batch_size = 1

    train_dataset_list = []
    val_dataset_list = []
    mask_dir_by_sample = {}
    supervised = all(dataset["masks"] is not None for dataset in datasets.value)
    training_enabled = supervised and not test_run
    split_manifest = {
        "config": {
            "validation_split": validation_split,
            "test_split": test_split,
            "split_seed": validation_seed,
        },
        "datasets": [],
    }

    util = MaskUtil() if do_post_process else None

    for dataset_idx, dataset in enumerate(datasets.value):
        image_folder = ImageFolder(root / dataset["images"])
        samples = [(Path(path), label) for path, label in image_folder.samples]
        mask_dir = root / dataset["masks"] if dataset["masks"] is not None else None

        for sample_path, _ in samples:
            mask_dir_by_sample[str(sample_path)] = mask_dir

        train_forgery_dataset = ForgeryDataset(
            samples=samples,
            mask_dir=mask_dir,
            size=image_size,
            transform=transform,
            return_path=False,
        )

        if supervised and (validation_split > 0.0 or test_split > 0.0):
            val_forgery_dataset = ForgeryDataset(
                samples=samples,
                mask_dir=mask_dir,
                size=image_size,
                transform=transform,
                return_path=True,
            )
            train_indices, val_indices, test_indices = split_indices_by_label_three_way(
                samples,
                validation_split=validation_split,
                test_split=test_split,
                seed=validation_seed + dataset_idx,
            )
            train_dataset_list.append(Subset(train_forgery_dataset, train_indices))
            if val_indices:
                val_dataset_list.append(Subset(val_forgery_dataset, val_indices))
            split_manifest["datasets"].append(
                {
                    "dataset_name": Path(dataset["images"]).name,
                    "mask_dir": str(mask_dir) if mask_dir is not None else None,
                    "seed": validation_seed + dataset_idx,
                    "train_count": len(train_indices),
                    "val_count": len(val_indices),
                    "test_count": len(test_indices),
                    "train_samples": [str(samples[idx][0]) for idx in train_indices],
                    "val_samples": [str(samples[idx][0]) for idx in val_indices],
                    "test_samples": [str(samples[idx][0]) for idx in test_indices],
                }
            )
        else:
            train_dataset_list.append(train_forgery_dataset)
            split_manifest["datasets"].append(
                {
                    "dataset_name": Path(dataset["images"]).name,
                    "mask_dir": str(mask_dir) if mask_dir is not None else None,
                    "seed": validation_seed + dataset_idx,
                    "train_count": len(samples),
                    "val_count": 0,
                    "test_count": 0,
                    "train_samples": [str(sample_path) for sample_path, _ in samples],
                    "val_samples": [],
                    "test_samples": [],
                }
            )

    train_dataset = combine_datasets(train_dataset_list)
    val_dataset = combine_datasets(val_dataset_list)
    split_manifest["summary"] = {
        "train_total": int(sum(entry["train_count"] for entry in split_manifest["datasets"])),
        "val_total": int(sum(entry["val_count"] for entry in split_manifest["datasets"])),
        "test_total": int(sum(entry["test_count"] for entry in split_manifest["datasets"])),
    }
    split_manifest_path, heldout_test_path = write_split_artifacts(output_dir, split_manifest)
    print(f"[pipeline] Wrote split manifest to {split_manifest_path}")
    print(f"[pipeline] Wrote held-out test sample list to {heldout_test_path}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
    split_summary = split_manifest["summary"]
    print(format_split_summary(split_summary, validation_seed))

    print("[pipeline] Loading PatchMatch feature branch...")
    pm_backbone, pyramid_zm = build_patchmatch_feature_branch(device)
    print("[pipeline] Loading SEUNet feature branch...")
    dino_extractor = build_seunet_feature_branch(
        device,
        feature_backbone=feature_backbone,
        dino_model_name=dino_model_name,
        separate_transforms=separate_transforms,
        use_dino_transform=use_dino_transform,
    )

    dlf_decoder = None
    se_model = None
    optimizer = None

    batch_counter = 0
    run_start = time.perf_counter()
    eta_batch_times = []
    eta_printed = False

    print('Starting training...')

    for epoch_idx in range(resume_epoch, epochs):
        epoch_start = time.perf_counter()
        train_accumulator = initialize_metric_accumulator()
        epoch_loss_steps = 0

        if optimizer is not None:
            set_frozen_feature_branch_modes(pm_backbone, pyramid_zm, dino_extractor)
            set_trainable_head_modes(dlf_decoder, se_model, training=True)

        for batch_idx, (images, masks, _labels) in enumerate(train_loader, start=1):
            batch_start = time.perf_counter()
            images = images.to(device)
            masks = masks.to(device=device, dtype=torch.float32)

            if supervised:
                mask_min = int(masks.min().item())
                mask_max = int(masks.max().item())
                if mask_min < 0 or mask_max > 1:
                    raise ValueError(
                        f"Dice loss expects binary target values in [0, 1], got range [{mask_min}, {mask_max}] "
                        f"at epoch {epoch_idx + 1}, batch {batch_idx}."
                    )

            collect_localization_stats = log_every > 0 and batch_idx % log_every == 0
            cnn_errors, zernike_errors, cnn_branch_result, zernike_branch_result, dino_features, localization_stats = extract_localization_inputs(
                images=images,
                pm_backbone=pm_backbone,
                pyramid_zm=pyramid_zm,
                dino_extractor=dino_extractor,
                separate_transforms=separate_transforms,
                cnn_feature_norm=cnn_feature_norm,
                pm_random_window=pm_random_window,
                pm_iters=pm_iters,
                pm_beta=pm_beta,
                pm_hard_selection=pm_hard_selection,
                pm_use_non_local=pm_use_non_local,
                pm_non_local_limit=pm_non_local_limit,
                pm_flat_threshold=pm_flat_threshold,
                pm_margin_threshold=pm_margin_threshold,
                pm_topk=pm_topk,
                pm_reduced_precision=pm_reduced_precision,
                localization_resolution=localization_resolution,
                dlf_error_scaling=dlf_error_scaling,
                collect_stats=collect_localization_stats,
            )

            # The trainable heads still depend on the first-batch channel counts, so we
            # initialize the PatchMatch decoder and SEUNet head lazily and restore once.
            heads_just_initialized = False
            if dlf_decoder is None:
                dlf_decoder = build_patchmatch_head(cnn_errors, device)
                heads_just_initialized = True
            if se_model is None:
                se_model = build_seunet_head(dino_features, device)
                heads_just_initialized = True
            if heads_just_initialized:
                set_trainable_head_modes(dlf_decoder, se_model, training=training_enabled)
                if training_enabled:
                    optimizer = build_localization_optimizer(dlf_decoder, se_model, learning_rate)
                checkpoint = restore_training_state(
                    checkpoint=checkpoint,
                    pm_backbone=pm_backbone,
                    dino_extractor=dino_extractor,
                    dlf_decoder=dlf_decoder,
                    se_model=se_model,
                    optimizer=optimizer,
                    learning_rate=learning_rate,
                )

            refined_mask, target_map, dlf_map = decode_and_refine_masks(
                images=images,
                cnn_error_maps=cnn_errors,
                zernike_error_maps=zernike_errors,
                cnn_branch_result=cnn_branch_result,
                zernike_branch_result=zernike_branch_result,
                dlf_decoder=dlf_decoder,
                se_model=se_model,
                dino_features=dino_features,
                output_size=images.shape[-2:],
            )

            if test_run:
                display_image(images[0], masks[0])
                display_pixel_offsets(cnn_branch_result.offsets[0], zernike_branch_result.offsets[0], images[0])
                mask_preds = post_process_predictions(
                    refined_mask,
                    util,
                    do_post_process=do_post_process,
                    post_process_threshold=post_process_threshold,
                    post_process_confident_threshold=post_process_confident_threshold,
                    post_process_min_component_area=post_process_min_component_area,
                    post_process_smooth_probabilities=post_process_smooth_probabilities,
                    post_process_fill_holes=post_process_fill_holes,
                    post_process_apply_closing=post_process_apply_closing,
                )
                display_image(images[0], (mask_preds[0]))
                return

            if optimizer is not None:
                optimizer.zero_grad()
                loss_terms = localization_loss_terms(
                    refined_mask,
                    target_map,
                    dlf_map,
                    masks,
                    mprime_loss_weight=mprime_loss_weight,
                    empty_target_penalty_weight=empty_target_penalty_weight,
                )
                loss = loss_terms[0]
                branch_stats = summarize_branch_activity(dlf_map, target_map)
                loss.backward()
                optimizer.step()

                update_metric_accumulator(train_accumulator, loss_terms, branch_stats)
                epoch_loss_steps += 1

                if log_every > 0 and batch_idx % log_every == 0:
                    print(
                        format_train_batch_message(
                            epoch_idx=epoch_idx,
                            epochs=epochs,
                            batch_idx=batch_idx,
                            total_batches=len(train_loader),
                            batch_summary=summarize_metric_step(loss_terms, branch_stats),
                            batch_seconds=time.perf_counter() - batch_start,
                            mprime_loss_weight=mprime_loss_weight,
                            empty_target_penalty_weight=empty_target_penalty_weight,
                            localization_stats=localization_stats,
                        )
                    )

                if epoch_idx == resume_epoch and batch_idx > 5:
                    eta_batch_times.append(time.perf_counter() - batch_start)
                    if not eta_printed and len(eta_batch_times) >= 5:
                        avg_batch_s = sum(eta_batch_times) / len(eta_batch_times)
                        est_train_total_s = avg_batch_s * epochs * len(train_loader)
                        print(
                            f"[pipeline] Warmed-up train-only estimate: "
                            f"{est_train_total_s / 60:.2f} minutes or {est_train_total_s / 3600:.2f} hours"
                        )
                        eta_printed = True

            if save_predictions:
                mask_preds = post_process_predictions(
                    refined_mask,
                    util,
                    do_post_process=do_post_process,
                    post_process_threshold=post_process_threshold,
                    post_process_confident_threshold=post_process_confident_threshold,
                    post_process_min_component_area=post_process_min_component_area,
                    post_process_smooth_probabilities=post_process_smooth_probabilities,
                    post_process_fill_holes=post_process_fill_holes,
                    post_process_apply_closing=post_process_apply_closing,
                )
                save_prediction_batch(predictions_dir, epoch_idx, batch_counter + 1, mask_preds)

            batch_counter += 1

        val_metrics = None
        if val_loader is not None and dlf_decoder is not None and se_model is not None:
            set_frozen_feature_branch_modes(pm_backbone, pyramid_zm, dino_extractor)
            set_trainable_head_modes(dlf_decoder, se_model, training=False)
            val_accumulator = initialize_metric_accumulator()
            val_loss_steps = 0
            val_counts = initialize_segmentation_counts()
            val_instance_tracker = initialize_instance_metric_tracker()

            with torch.no_grad():
                for images, masks, _labels, image_paths in val_loader:
                    images = images.to(device)
                    masks = masks.to(device=device, dtype=torch.float32)

                    cnn_errors, zernike_errors, cnn_branch_result, zernike_branch_result, dino_features, _ = extract_localization_inputs(
                        images=images,
                        pm_backbone=pm_backbone,
                        pyramid_zm=pyramid_zm,
                        dino_extractor=dino_extractor,
                        separate_transforms=separate_transforms,
                        cnn_feature_norm=cnn_feature_norm,
                        pm_random_window=pm_random_window,
                        pm_iters=pm_iters,
                        pm_beta=pm_beta,
                        pm_hard_selection=pm_hard_selection,
                        pm_use_non_local=pm_use_non_local,
                        pm_non_local_limit=pm_non_local_limit,
                        pm_flat_threshold=pm_flat_threshold,
                        pm_margin_threshold=pm_margin_threshold,
                        pm_topk=pm_topk,
                        pm_reduced_precision=pm_reduced_precision,
                        localization_resolution=localization_resolution,
                        dlf_error_scaling=dlf_error_scaling,
                    )
                    refined_mask, target_map, dlf_map = decode_and_refine_masks(
                        images=images,
                        cnn_error_maps=cnn_errors,
                        zernike_error_maps=zernike_errors,
                        cnn_branch_result=cnn_branch_result,
                        zernike_branch_result=zernike_branch_result,
                        dlf_decoder=dlf_decoder,
                        se_model=se_model,
                        dino_features=dino_features,
                        output_size=images.shape[-2:],
                    )
                    val_loss_terms = localization_loss_terms(
                        refined_mask,
                        target_map,
                        dlf_map,
                        masks,
                        mprime_loss_weight=mprime_loss_weight,
                        empty_target_penalty_weight=empty_target_penalty_weight,
                    )
                    branch_stats = summarize_branch_activity(dlf_map, target_map)
                    update_metric_accumulator(val_accumulator, val_loss_terms, branch_stats)
                    val_loss_steps += 1

                    mask_preds = post_process_predictions(
                        refined_mask,
                        util,
                        do_post_process=do_post_process,
                        post_process_threshold=post_process_threshold,
                        post_process_confident_threshold=post_process_confident_threshold,
                        post_process_min_component_area=post_process_min_component_area,
                        post_process_smooth_probabilities=post_process_smooth_probabilities,
                        post_process_fill_holes=post_process_fill_holes,
                        post_process_apply_closing=post_process_apply_closing,
                    )

                    update_segmentation_counts(mask_preds, masks.long(), val_counts)

                    pred_masks_np = mask_preds.cpu().numpy().astype(np.uint8)
                    # Pixel metrics are already tracked above; the instance-level oF1 needs
                    # per-image connected components loaded from the original sample paths.
                    for pred_mask, image_path in zip(pred_masks_np, image_paths):
                        pred_instances = binary_mask_to_instances(pred_mask)
                        gt_instances = load_resized_gt_instances(
                            image_path,
                            mask_dir_by_sample=mask_dir_by_sample,
                            image_size=image_size,
                        )
                        image_of1 = optimal_f1_score(pred_instances, gt_instances)
                        pred_component_count = len(pred_instances)
                        gt_component_count = len(gt_instances)

                        update_instance_metric_tracker(
                            val_instance_tracker,
                            image_of1=image_of1,
                            pred_component_count=pred_component_count,
                            gt_component_count=gt_component_count,
                        )

            val_metrics = summarize_segmentation_counts(val_counts)
            # Validation combines batch-level losses with per-image instance metrics,
            # so we collapse both trackers here instead of hand-assembling the dict inline.
            val_summary = build_validation_summary(
                val_accumulator,
                val_loss_steps,
                val_metrics,
                val_instance_tracker,
            )
            print(
                format_validation_message(
                    epoch_idx=epoch_idx,
                    epochs=epochs,
                    val_summary=val_summary,
                    mprime_loss_weight=mprime_loss_weight,
                    empty_target_penalty_weight=empty_target_penalty_weight,
                )
            )

            if optimizer is not None:
                set_frozen_feature_branch_modes(pm_backbone, pyramid_zm, dino_extractor)
                set_trainable_head_modes(dlf_decoder, se_model, training=True)

        if epoch_loss_steps > 0:
            epoch_seconds = time.perf_counter() - epoch_start
            train_summary = average_metric_accumulator(train_accumulator, epoch_loss_steps)
            metrics = build_epoch_metrics(
                epoch_idx=epoch_idx,
                epoch_seconds=epoch_seconds,
                train_summary=train_summary,
                steps=epoch_loss_steps,
                val_summary=val_summary if val_metrics is not None else None,
            )

            print(
                format_train_epoch_message(
                    epoch_idx=epoch_idx,
                    epochs=epochs,
                    train_summary=train_summary,
                    epoch_seconds=epoch_seconds,
                    mprime_loss_weight=mprime_loss_weight,
                    empty_target_penalty_weight=empty_target_penalty_weight,
                )
            )
            append_metrics_log(output_dir, metrics)
            save_metrics_plot(output_dir)

            if optimizer is not None:
                checkpoint_score = val_summary["of1"] if val_metrics is not None else -train_summary["loss"]
                best_score = save_epoch_checkpoints(
                    checkpoint_path=checkpoint_path,
                    best_checkpoint_path=best_checkpoint_path,
                    epoch=epoch_idx + 1,
                    dlf_decoder=dlf_decoder,
                    se_model=se_model,
                    optimizer=optimizer,
                    checkpoint_score=checkpoint_score,
                    best_score=best_score,
                    dino_extractor=dino_extractor,
                    pm_backbone=pm_backbone,
                )

    save_metrics_plot(output_dir)
    print(f"Training completed. Total time: {time.perf_counter() - run_start:.2f}s")
