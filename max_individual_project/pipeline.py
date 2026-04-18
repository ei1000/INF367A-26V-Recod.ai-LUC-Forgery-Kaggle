from pathlib import Path
import time
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from dataset import (
    Datasets,
    ForgeryDataset,
    combine_datasets,
    resolve_data_root,
    resolve_image_transform,
    split_indices_by_label_three_way,
)
from feature_extractors.cnn_feature_extractor import (
    PretrainedBackboneExtractor,
    SingleScaleFeatureExtractor,
)
from feature_extractors.zernike_feature_extractor import PyramidZernikeExtractor, default_pq_list
from prediction.decoder import DLFDecoder
from prediction.localization import decode_and_refine_masks, extract_localization_inputs
from prediction.mask_metrics import (
    binary_mask_to_instances,
    load_resized_gt_instances,
    optimal_f1_score,
    summarize_segmentation_counts,
    update_segmentation_counts,
)
from prediction.pixelmaputil_mask import MaskUtil, post_process_mask_batch
from prediction.se_u_net import SEUNet
from training.checkpointing import ensure_output_dirs, load_module_state, save_checkpoint, save_prediction_batch
from training.losses import localization_loss_terms, summarize_branch_activity
from training.metrics_logging import append_metrics_log, save_metrics_plot
from training.optim import (
    set_optimizer_learning_rate,
)
from visualizer import display_image, display_pixel_offsets


def pipeline(
    datasets=Datasets.TRAIN,
    image_size=3000,
    epochs=1,
    test_run=False,
    feature_backbone="dino_single",
    use_dino_transform=False,
    batch_size=1,
    override_batch_size=False,
    dino_model_name="dinov2_vits14",
    dino_proj_dim=64,
    cnn_backbone="simple",
    cnn_feature_norm=True,
    separate_transforms=True,
    pm_iters=24,
    pm_beta=10.0,
    pm_hard_selection=True,
    pm_random_window=50,
    pm_use_non_local=True,
    pm_non_local_limit=25.0,
    pm_flat_threshold=0.15,
    pm_margin_threshold=0.10,
    pm_topk=1,
    pm_reduced_precision=True,
    dino_match_native_resolution=False,
    localization_resolution="image",
    train_feature_backbone=False,
    feature_backbone_learning_rate=None,
    dino_finetune_blocks=0,
    log_every=10,
    output_dir="artifacts",
    checkpoint_name="latest.pt",
    resume=True,
    save_predictions=False,
    validation_split=0.0,
    test_split=0.0,
    validation_seed=42,
    learning_rate=1e-3,
    mprime_loss_weight=0.5,
    empty_target_penalty_weight=0.0,
    dlf_error_scaling="log1p",
    do_post_process=True,
    post_process_threshold=0.5,
    post_process_confident_threshold=None,
    post_process_smooth_probabilities=False,
    post_process_fill_holes=True,
    post_process_apply_closing=False,
    post_process_min_component_area=0,
):
    """Train or inspect the copy-move localization pipeline.

    Current architecture:
    - frozen ResNet18 PatchMatch branch
    - frozen multi-scale Zernike PatchMatch branch
    - frozen DINO features feeding an SEUNet refinement head
    """

    print('[pipeline] Initializing training loop and datasets...')
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if train_feature_backbone:
        print("[pipeline] Keeping DINO and ResNet backbones frozen; train_feature_backbone=True is ignored in this setup.")
    if dino_proj_dim is not None:
        print("[pipeline] Ignoring dino_proj_dim so the SEUNet branch consumes raw frozen DINO features.")
    if dino_match_native_resolution:
        print("[pipeline] Ignoring dino_match_native_resolution because localization now always runs on the image grid.")
    if feature_backbone_learning_rate is not None:
        print("[pipeline] Ignoring feature_backbone_learning_rate because all feature extractors are frozen.")
    if dino_finetune_blocks:
        print("[pipeline] Ignoring dino_finetune_blocks because the DINO branch stays frozen in this pipeline.")
    output_dir, checkpoints_dir, predictions_dir = ensure_output_dirs(output_dir)
    checkpoint_path = checkpoints_dir / checkpoint_name
    best_checkpoint_path = checkpoints_dir / "best.pt"
    checkpoint = None
    resume_epoch = 0
    best_score = None

    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        resume_epoch = int(checkpoint.get("epoch", 0))
        best_score = checkpoint.get("best_score", checkpoint.get("best_loss"))
        print(f"[pipeline] Resuming from checkpoint: {checkpoint_path}")

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
    split_manifest_path = output_dir / "split_manifest.json"
    split_manifest_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    print(f"[pipeline] Wrote split manifest to {split_manifest_path}")
    heldout_test_samples = [sample for entry in split_manifest["datasets"] for sample in entry["test_samples"]]
    heldout_test_path = output_dir / "heldout_test_samples.txt"
    heldout_test_path.write_text("\n".join(heldout_test_samples), encoding="utf-8")
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
    print(
        "[pipeline] Split summary: "
        f"train={split_summary['train_total']} "
        f"val={split_summary['val_total']} "
        f"test={split_summary['test_total']} "
        f"(seed={validation_seed})"
    )

    print('Loading models and creating ZernikeFeatures...')
    from feature_extractors.dino_feature_extractor import PyramidDinoFeatureExtractor, SingleScaleDinoFeatureExtractor

    # `feature_backbone` now selects the frozen DINO feature extractor used by the refinement branch.
    dino_extractor_cls = PyramidDinoFeatureExtractor if feature_backbone == "dino" else SingleScaleDinoFeatureExtractor
    dino_extractor = dino_extractor_cls(
        model_name=dino_model_name,
        freeze=True,
        finetune_blocks=0,
        normalize_input=True if separate_transforms else not use_dino_transform,
        proj_dim=None,
        upsample_to_input=False,
    ).to(device)
    pm_backbone = SingleScaleFeatureExtractor(
        backbone=PretrainedBackboneExtractor(
            model_name="resnet18",
            out_dim=32,
            freeze=True,
        ),
        upsample_to_input=True,
    ).to(device)

    pq_list = default_pq_list(max_order=5)
    pyramid_zm = PyramidZernikeExtractor(pq_list, kernel_size=13).to(device)
    pm_backbone.eval()
    dino_extractor.eval()
    pyramid_zm.eval()

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
        epoch_loss_sum = 0.0
        epoch_ldfm_sum = 0.0
        epoch_lmrd_sum = 0.0
        epoch_mprime_loss_sum = 0.0
        epoch_empty_target_loss_sum = 0.0
        epoch_empty_refined_loss_sum = 0.0
        epoch_empty_target_map_loss_sum = 0.0
        epoch_empty_mprime_map_loss_sum = 0.0
        epoch_mprime_positive_rate_sum = 0.0
        epoch_mprime_wins_rate_sum = 0.0
        epoch_target_positive_rate_sum = 0.0
        epoch_loss_steps = 0

        if optimizer is not None:
            pm_backbone.eval()
            dino_extractor.eval()
            pyramid_zm.eval()
            dlf_decoder.train()
            se_model.train()

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

            if dlf_decoder is None or se_model is None:
                dlf_decoder = DLFDecoder(
                    num_error_maps=cnn_errors.shape[1],
                ).to(device)
                se_model = SEUNet(
                    in_channels=dino_features.shape[1],
                    out_channels=1,
                    final_activation="sigmoid",
                ).to(device)
                if test_run or not supervised:
                    dlf_decoder.eval()
                    se_model.eval()
                else:
                    dlf_decoder.train()
                    se_model.train()
                    optimizer = torch.optim.Adam(
                        [
                            {"params": list(dlf_decoder.parameters()), "lr": learning_rate, "name": "dlf_decoder"},
                            {"params": list(se_model.parameters()), "lr": learning_rate, "name": "se_model"},
                        ],
                        lr=learning_rate,
                    )

                if checkpoint is not None:
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
                    checkpoint = None

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
                mask_probs = refined_mask.squeeze(1)
                mask_preds = (
                    post_process_mask_batch(
                        mask_probs,
                        util,
                        threshold=post_process_threshold,
                        confident_threshold=post_process_confident_threshold,
                        min_component_area=post_process_min_component_area,
                        smooth_probabilities=post_process_smooth_probabilities,
                        fill_holes=post_process_fill_holes,
                        apply_closing=post_process_apply_closing,
                    )
                    if do_post_process and util is not None
                    else (mask_probs >= 0.5).long()
                )
                display_image(images[0], (mask_preds[0]))
                return

            if optimizer is not None:
                optimizer.zero_grad()
                (
                    loss,
                    ldfm,
                    lmrd,
                    mprime_loss,
                    empty_target_loss,
                    empty_refined_loss,
                    empty_target_map_loss,
                    empty_mprime_map_loss,
                ) = localization_loss_terms(
                    refined_mask,
                    target_map,
                    dlf_map,
                    masks,
                    mprime_loss_weight=mprime_loss_weight,
                    empty_target_penalty_weight=empty_target_penalty_weight,
                )
                branch_stats = summarize_branch_activity(dlf_map, target_map)
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                ldfm_value = ldfm.item()
                lmrd_value = lmrd.item()
                mprime_loss_value = mprime_loss.item()
                empty_target_loss_value = empty_target_loss.item()
                empty_refined_loss_value = empty_refined_loss.item()
                empty_target_map_loss_value = empty_target_map_loss.item()
                empty_mprime_map_loss_value = empty_mprime_map_loss.item()
                epoch_loss_sum += loss_value
                epoch_ldfm_sum += ldfm_value
                epoch_lmrd_sum += lmrd_value
                epoch_mprime_loss_sum += mprime_loss_value
                epoch_empty_target_loss_sum += empty_target_loss_value
                epoch_empty_refined_loss_sum += empty_refined_loss_value
                epoch_empty_target_map_loss_sum += empty_target_map_loss_value
                epoch_empty_mprime_map_loss_sum += empty_mprime_map_loss_value
                epoch_mprime_positive_rate_sum += branch_stats["mprime_positive_rate"]
                epoch_mprime_wins_rate_sum += branch_stats["mprime_wins_rate"]
                epoch_target_positive_rate_sum += branch_stats["target_positive_rate"]
                epoch_loss_steps += 1

                if log_every > 0 and batch_idx % log_every == 0:
                    localization_message = ""
                    if localization_stats is not None:
                        localization_message = (
                            f" feat: {localization_stats['feature_time_s']:.2f}s"
                            f" pm: {localization_stats['patchmatch_time_s']:.2f}s"
                            f" dlf: {localization_stats['dlf_time_s']:.2f}s"
                        )
                        peak_memory_mb = localization_stats.get("localization_peak_memory_mb")
                        if peak_memory_mb is not None:
                            localization_message += f" loc_peak: {peak_memory_mb:.0f}MB"
                    print(
                        f"[epoch {epoch_idx + 1}/{epochs}] "
                        f"batch {batch_idx}/{len(train_loader)} "
                        f"loss: {loss_value:.4f} "
                        f"(ldfm={ldfm_value:.4f}, lmrd={lmrd_value:.4f}, mprime={mprime_loss_value:.4f}, "
                        f"empty={empty_target_loss_value:.4f}[ref={empty_refined_loss_value:.4f}, "
                        f"se={empty_target_map_loss_value:.4f}, dlf={empty_mprime_map_loss_value:.4f}], "
                        f"lambda={mprime_loss_weight:.2f}, empty_lambda={empty_target_penalty_weight:.2f}) "
                        f"mprime_pos: {branch_stats['mprime_positive_rate']:.4%} "
                        f"mprime_wins: {branch_stats['mprime_wins_rate']:.4%} "
                        f"target_pos: {branch_stats['target_positive_rate']:.4%} "
                        f"time spent: {(time.perf_counter() - batch_start):.2f}"
                        f"{localization_message}"
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
                mask_probs = refined_mask.squeeze(1)
                mask_preds = (
                    post_process_mask_batch(
                        mask_probs,
                        util,
                        threshold=post_process_threshold,
                        confident_threshold=post_process_confident_threshold,
                        min_component_area=post_process_min_component_area,
                        smooth_probabilities=post_process_smooth_probabilities,
                        fill_holes=post_process_fill_holes,
                        apply_closing=post_process_apply_closing,
                    )
                    if do_post_process and util is not None
                    else (mask_probs >= 0.5).long()
                )
                save_prediction_batch(predictions_dir, epoch_idx, batch_counter + 1, mask_preds)

            batch_counter += 1

        val_metrics = None
        if val_loader is not None and dlf_decoder is not None and se_model is not None:
            pm_backbone.eval()
            dino_extractor.eval()
            pyramid_zm.eval()
            dlf_decoder.eval()
            se_model.eval()
            val_loss_sum = 0.0
            val_ldfm_sum = 0.0
            val_lmrd_sum = 0.0
            val_mprime_loss_sum = 0.0
            val_empty_target_loss_sum = 0.0
            val_empty_refined_loss_sum = 0.0
            val_empty_target_map_loss_sum = 0.0
            val_empty_mprime_map_loss_sum = 0.0
            val_mprime_positive_rate_sum = 0.0
            val_mprime_wins_rate_sum = 0.0
            val_target_positive_rate_sum = 0.0
            val_loss_steps = 0
            val_counts = {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "pred_pos": 0,
                "mask_pos": 0,
                "pixels": 0,
            }
            val_of1_sum = 0.0
            val_images = 0
            val_pred_components_sum = 0
            val_authentic_of1_sum = 0.0
            val_authentic_images = 0
            val_authentic_empty_pred_count = 0
            val_authentic_pred_components_sum = 0
            val_forged_of1_sum = 0.0
            val_forged_images = 0
            val_forged_pred_components_sum = 0
            val_forged_gt_components_sum = 0

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


                    (
                        val_loss,
                        val_ldfm,
                        val_lmrd,
                        val_mprime_loss,
                        val_empty_target_loss,
                        val_empty_refined_loss,
                        val_empty_target_map_loss,
                        val_empty_mprime_map_loss,
                    ) = localization_loss_terms(
                        refined_mask,
                        target_map,
                        dlf_map,
                        masks,
                        mprime_loss_weight=mprime_loss_weight,
                        empty_target_penalty_weight=empty_target_penalty_weight,
                    )
                    branch_stats = summarize_branch_activity(dlf_map, target_map)
                    val_loss_sum += val_loss.item()
                    val_ldfm_sum += val_ldfm.item()
                    val_lmrd_sum += val_lmrd.item()
                    val_mprime_loss_sum += val_mprime_loss.item()
                    val_empty_target_loss_sum += val_empty_target_loss.item()
                    val_empty_refined_loss_sum += val_empty_refined_loss.item()
                    val_empty_target_map_loss_sum += val_empty_target_map_loss.item()
                    val_empty_mprime_map_loss_sum += val_empty_mprime_map_loss.item()
                    val_mprime_positive_rate_sum += branch_stats["mprime_positive_rate"]
                    val_mprime_wins_rate_sum += branch_stats["mprime_wins_rate"]
                    val_target_positive_rate_sum += branch_stats["target_positive_rate"]
                    val_loss_steps += 1

                    mask_probs = refined_mask.squeeze(1)
                    mask_preds = (
                        post_process_mask_batch(
                            mask_probs,
                            util,
                            threshold=post_process_threshold,
                            confident_threshold=post_process_confident_threshold,
                            min_component_area=post_process_min_component_area,
                            smooth_probabilities=post_process_smooth_probabilities,
                            fill_holes=post_process_fill_holes,
                            apply_closing=post_process_apply_closing,
                        )
                        if do_post_process and util is not None
                        else (mask_probs >= 0.5).long()
                    )

                    update_segmentation_counts(mask_preds, masks.long(), val_counts)

                    pred_masks_np = mask_preds.cpu().numpy().astype(np.uint8)
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

                        val_of1_sum += image_of1
                        val_images += 1
                        val_pred_components_sum += pred_component_count

                        if gt_component_count == 0:
                            val_authentic_of1_sum += image_of1
                            val_authentic_images += 1
                            val_authentic_pred_components_sum += pred_component_count
                            if pred_component_count == 0:
                                val_authentic_empty_pred_count += 1
                        else:
                            val_forged_of1_sum += image_of1
                            val_forged_images += 1
                            val_forged_pred_components_sum += pred_component_count
                            val_forged_gt_components_sum += gt_component_count

            val_metrics = summarize_segmentation_counts(val_counts)
            val_mean_loss = val_loss_sum / max(val_loss_steps, 1)
            val_mean_ldfm = val_ldfm_sum / max(val_loss_steps, 1)
            val_mean_lmrd = val_lmrd_sum / max(val_loss_steps, 1)
            val_mean_mprime_loss = val_mprime_loss_sum / max(val_loss_steps, 1)
            val_mean_empty_target_loss = val_empty_target_loss_sum / max(val_loss_steps, 1)
            val_mean_empty_refined_loss = val_empty_refined_loss_sum / max(val_loss_steps, 1)
            val_mean_empty_target_map_loss = val_empty_target_map_loss_sum / max(val_loss_steps, 1)
            val_mean_empty_mprime_map_loss = val_empty_mprime_map_loss_sum / max(val_loss_steps, 1)
            val_mean_mprime_positive_rate = val_mprime_positive_rate_sum / max(val_loss_steps, 1)
            val_mean_mprime_wins_rate = val_mprime_wins_rate_sum / max(val_loss_steps, 1)
            val_mean_target_positive_rate = val_target_positive_rate_sum / max(val_loss_steps, 1)
            val_mean_of1 = val_of1_sum / max(val_images, 1)
            val_mean_pred_components = val_pred_components_sum / max(val_images, 1)
            val_mean_authentic_of1 = val_authentic_of1_sum / max(val_authentic_images, 1)
            val_mean_authentic_pred_components = val_authentic_pred_components_sum / max(val_authentic_images, 1)
            val_authentic_empty_pred_rate = val_authentic_empty_pred_count / max(val_authentic_images, 1)
            val_mean_forged_of1 = val_forged_of1_sum / max(val_forged_images, 1)
            val_mean_forged_pred_components = val_forged_pred_components_sum / max(val_forged_images, 1)
            val_mean_forged_gt_components = val_forged_gt_components_sum / max(val_forged_images, 1)
            print(
                f"[epoch {epoch_idx + 1}/{epochs}] "
                f"val_loss: {val_mean_loss:.4f} "
                f"(ldfm={val_mean_ldfm:.4f}, lmrd={val_mean_lmrd:.4f}, mprime={val_mean_mprime_loss:.4f}, "
                f"empty={val_mean_empty_target_loss:.4f}[ref={val_mean_empty_refined_loss:.4f}, "
                f"se={val_mean_empty_target_map_loss:.4f}, dlf={val_mean_empty_mprime_map_loss:.4f}], "
                f"lambda={mprime_loss_weight:.2f}, empty_lambda={empty_target_penalty_weight:.2f}) "
                f"val_oF1: {val_mean_of1:.4f} "
                f"val_pred_pos: {val_metrics['pred_positive_rate']:.4%} "
                f"val_mprime_pos: {val_mean_mprime_positive_rate:.4%} "
                f"val_mprime_wins: {val_mean_mprime_wins_rate:.4%} "
                f"val_target_pos: {val_mean_target_positive_rate:.4%} "
                f"pred_components/img: {val_mean_pred_components:.2f} "
                f"auth_oF1: {val_mean_authentic_of1:.4f} "
                f"auth_empty_pred: {val_authentic_empty_pred_rate:.2%} "
                f"auth_components/img: {val_mean_authentic_pred_components:.2f} "
                f"forged_oF1: {val_mean_forged_of1:.4f} "
                f"forged_components/img: {val_mean_forged_pred_components:.2f} "
                f"forged_gt_components/img: {val_mean_forged_gt_components:.2f}"
            )

            if optimizer is not None:
                pm_backbone.eval()
                dino_extractor.eval()
                pyramid_zm.eval()
                dlf_decoder.train()
                se_model.train()

        if epoch_loss_steps > 0:
            mean_loss = epoch_loss_sum / epoch_loss_steps
            mean_ldfm = epoch_ldfm_sum / epoch_loss_steps
            mean_lmrd = epoch_lmrd_sum / epoch_loss_steps
            mean_mprime_loss = epoch_mprime_loss_sum / epoch_loss_steps
            mean_empty_target_loss = epoch_empty_target_loss_sum / epoch_loss_steps
            mean_empty_refined_loss = epoch_empty_refined_loss_sum / epoch_loss_steps
            mean_empty_target_map_loss = epoch_empty_target_map_loss_sum / epoch_loss_steps
            mean_empty_mprime_map_loss = epoch_empty_mprime_map_loss_sum / epoch_loss_steps
            mean_mprime_positive_rate = epoch_mprime_positive_rate_sum / epoch_loss_steps
            mean_mprime_wins_rate = epoch_mprime_wins_rate_sum / epoch_loss_steps
            mean_target_positive_rate = epoch_target_positive_rate_sum / epoch_loss_steps
            metrics = {
                "epoch": epoch_idx + 1,
                "train_loss": mean_loss,
                "train_ldfm": mean_ldfm,
                "train_lmrd": mean_lmrd,
                "train_mprime_loss": mean_mprime_loss,
                "train_empty_target_loss": mean_empty_target_loss,
                "train_empty_refined_loss": mean_empty_refined_loss,
                "train_empty_target_map_loss": mean_empty_target_map_loss,
                "train_empty_mprime_map_loss": mean_empty_mprime_map_loss,
                "train_mprime_positive_rate": mean_mprime_positive_rate,
                "train_mprime_wins_rate": mean_mprime_wins_rate,
                "train_target_positive_rate": mean_target_positive_rate,
                "steps": epoch_loss_steps,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }

            if val_metrics is not None:
                metrics["val_loss"] = val_mean_loss
                metrics["val_ldfm"] = val_mean_ldfm
                metrics["val_lmrd"] = val_mean_lmrd
                metrics["val_mprime_loss"] = val_mean_mprime_loss
                metrics["val_empty_target_loss"] = val_mean_empty_target_loss
                metrics["val_empty_refined_loss"] = val_mean_empty_refined_loss
                metrics["val_empty_target_map_loss"] = val_mean_empty_target_map_loss
                metrics["val_empty_mprime_map_loss"] = val_mean_empty_mprime_map_loss
                metrics["val_mprime_positive_rate"] = val_mean_mprime_positive_rate
                metrics["val_mprime_wins_rate"] = val_mean_mprime_wins_rate
                metrics["val_target_positive_rate"] = val_mean_target_positive_rate
                metrics["val_of1"] = val_mean_of1
                metrics["val_pred_components_per_image"] = val_mean_pred_components
                metrics["val_authentic_of1"] = val_mean_authentic_of1
                metrics["val_authentic_empty_pred_rate"] = val_authentic_empty_pred_rate
                metrics["val_authentic_pred_components_per_image"] = val_mean_authentic_pred_components
                metrics["val_forged_of1"] = val_mean_forged_of1
                metrics["val_forged_pred_components_per_image"] = val_mean_forged_pred_components
                metrics["val_forged_gt_components_per_image"] = val_mean_forged_gt_components
                metrics["val_iou"] = val_metrics["iou"]
                metrics["val_dice"] = val_metrics["dice"]
                metrics["val_pred_positive_rate"] = val_metrics["pred_positive_rate"]
                metrics["val_mask_positive_rate"] = val_metrics["mask_positive_rate"]

            print(
                f"[epoch {epoch_idx + 1}/{epochs}] "
                f"train_loss: {mean_loss:.4f} "
                f"(ldfm={mean_ldfm:.4f}, lmrd={mean_lmrd:.4f}, mprime={mean_mprime_loss:.4f}, "
                f"empty={mean_empty_target_loss:.4f}[ref={mean_empty_refined_loss:.4f}, "
                f"se={mean_empty_target_map_loss:.4f}, dlf={mean_empty_mprime_map_loss:.4f}], "
                f"lambda={mprime_loss_weight:.2f}, empty_lambda={empty_target_penalty_weight:.2f}) "
                f"mprime_pos: {mean_mprime_positive_rate:.4%} "
                f"mprime_wins: {mean_mprime_wins_rate:.4%} "
                f"target_pos: {mean_target_positive_rate:.4%} "
                f"completed in: {metrics['epoch_seconds']:.2f}s"
            )
            append_metrics_log(output_dir, metrics)
            save_metrics_plot(output_dir)

            if optimizer is not None:
                checkpoint_score = val_mean_of1 if val_metrics is not None else -mean_loss
                save_checkpoint(
                    checkpoint_path,
                    epoch=epoch_idx + 1,
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
                        epoch=epoch_idx + 1,
                        dlf_decoder=dlf_decoder,
                        se_model=se_model,
                        optimizer=optimizer,
                        best_score=best_score,
                        dino_extractor=dino_extractor,
                        pm_backbone=pm_backbone,
                    )

    save_metrics_plot(output_dir)
    print(f"Training completed. Total time: {time.perf_counter() - run_start:.2f}s")
