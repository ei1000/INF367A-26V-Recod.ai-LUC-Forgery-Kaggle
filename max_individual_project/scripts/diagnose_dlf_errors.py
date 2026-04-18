from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import (
    Datasets,
    ForgeryDataset,
    combine_datasets,
    resolve_data_root,
    resolve_image_transform,
    split_indices_by_label,
)
from feature_extractors.cnn_feature_extractor import PretrainedBackboneExtractor, SingleScaleFeatureExtractor
from feature_extractors.dino_feature_extractor import PyramidDinoFeatureExtractor, SingleScaleDinoFeatureExtractor
from feature_extractors.zernike_feature_extractor import PyramidZernikeExtractor, default_pq_list
from prediction.localization import decode_and_refine_masks, extract_localization_inputs, normalize_dlf_error_maps
from prediction.decoder import DLFDecoder
from prediction.se_u_net import SEUNet
from training.checkpointing import load_module_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect DLF error-map magnitudes and branch behavior.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT.parent / "artifacts" / "cnn_pretrained_frozen_run" / "checkpoints" / "latest.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT.parent / "artifacts" / "cnn_pretrained_frozen_run" / "dlf_diagnostics",
    )
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--validation-seed", type=int, default=42)
    parser.add_argument("--feature-backbone", choices=("cnn", "dino", "dino_single"), default="cnn")
    parser.add_argument("--use-dino-transform", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dino-model-name", default="dinov2_vits14")
    parser.add_argument("--dino-proj-dim", type=int, default=64)
    parser.add_argument("--dino-finetune-blocks", type=int, default=0)
    parser.add_argument("--cnn-backbone", choices=("simple", "pretrained"), default="pretrained")
    parser.add_argument("--cnn-pretrained-model", default="vgg16_bn")
    parser.add_argument("--cnn-feature-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--separate-transforms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pm-iters", type=int, default=20)
    parser.add_argument("--pm-beta", type=float, default=10.0)
    parser.add_argument("--pm-hard-selection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pm-random-window", type=int, default=50)
    parser.add_argument("--pm-use-non-local", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pm-non-local-limit", type=float, default=25.0)
    parser.add_argument("--pm-flat-threshold", type=float, default=0.15)
    parser.add_argument("--pm-margin-threshold", type=float, default=0.10)
    parser.add_argument("--pm-topk", type=int, default=1)
    parser.add_argument("--pm-reduced-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dino-match-native-resolution", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--localization-resolution", choices=("image", "feature_grid"), default="image")
    parser.add_argument("--current-scaling", choices=("none", "log1p", "zscore"), default="log1p")
    return parser.parse_args()


def choose_transform(args: argparse.Namespace):
    return resolve_image_transform(
        feature_backbone=args.feature_backbone,
        use_dino_transform=args.use_dino_transform,
        cnn_backbone=args.cnn_backbone,
        separate_transforms=args.separate_transforms,
    )


def build_validation_dataset(args: argparse.Namespace):
    root = resolve_data_root()
    transform = choose_transform(args)
    val_dataset_list = []

    for dataset_idx, dataset in enumerate(Datasets.ALL_TRAIN.value):
        image_folder = ImageFolder(root / dataset["images"])
        samples = [(Path(path), label) for path, label in image_folder.samples]
        mask_dir = root / dataset["masks"] if dataset["masks"] is not None else None
        val_forgery_dataset = ForgeryDataset(
            samples=samples,
            mask_dir=mask_dir,
            size=args.image_size,
            transform=transform,
            return_path=True,
        )
        _, val_indices = split_indices_by_label(
            samples,
            validation_split=args.validation_split,
            seed=args.validation_seed + dataset_idx,
        )
        if val_indices:
            val_dataset_list.append(Subset(val_forgery_dataset, val_indices))

    return combine_datasets(val_dataset_list)


def iter_subset_paths(dataset, base_offset: int = 0):
    if isinstance(dataset, ConcatDataset):
        running = base_offset
        for sub_dataset in dataset.datasets:
            yield from iter_subset_paths(sub_dataset, running)
            running += len(sub_dataset)
        return

    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        for local_idx, sample_idx in enumerate(dataset.indices):
            sample_path, _ = base_dataset.samples[sample_idx]
            yield base_offset + local_idx, Path(sample_path)
        return

    if isinstance(dataset, ForgeryDataset):
        for local_idx, (sample_path, _) in enumerate(dataset.samples):
            yield base_offset + local_idx, Path(sample_path)
        return

    raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def select_balanced_indices(dataset, samples_per_class: int) -> list[int]:
    authentic_indices = []
    forged_indices = []

    for idx, sample_path in iter_subset_paths(dataset):
        is_forged = "forged" in sample_path.parent.name
        target = forged_indices if is_forged else authentic_indices
        if len(target) < samples_per_class:
            target.append(idx)
        if len(authentic_indices) >= samples_per_class and len(forged_indices) >= samples_per_class:
            break

    selected = authentic_indices + forged_indices
    selected.sort()
    return selected


def build_feature_extractors(args: argparse.Namespace, device: torch.device):
    dino_extractor_cls = PyramidDinoFeatureExtractor if args.feature_backbone == "dino" else SingleScaleDinoFeatureExtractor
    dino_extractor = dino_extractor_cls(
        model_name=args.dino_model_name,
        freeze=True,
        finetune_blocks=0,
        normalize_input=True if args.separate_transforms else not args.use_dino_transform,
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
    return pm_backbone, dino_extractor, pyramid_zm


def summarize_array(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}

    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def roc_auc_score_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    scores = scores.astype(np.float64)
    pos_count = int(labels.sum())
    neg_count = int((1 - labels).sum())
    if pos_count == 0 or neg_count == 0:
        return 0.0

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_ranks = ranks[labels == 1]
    return float((pos_ranks.sum() - (pos_count * (pos_count + 1) / 2.0)) / (pos_count * neg_count))


def count_components(mask: np.ndarray) -> int:
    _, count = scipy.ndimage.label(mask.astype(bool))
    return int(count)


def topk_precision(score_map: np.ndarray, mask: np.ndarray) -> float:
    flat_scores = score_map.reshape(-1)
    flat_mask = mask.reshape(-1).astype(bool)
    k = int(flat_mask.sum())
    if k <= 0:
        return 0.0

    if k >= flat_scores.size:
        topk_idx = np.arange(flat_scores.size)
    else:
        topk_idx = np.argpartition(flat_scores, -k)[-k:]
    return float(flat_mask[topk_idx].mean())


def transform_variant(raw_errors: np.ndarray, variant: str) -> np.ndarray:
    error_tensor = torch.from_numpy(raw_errors)
    if variant in {"none", "log1p", "zscore"}:
        return normalize_dlf_error_maps(error_tensor, mode=variant).cpu().numpy()
    if variant == "sqrt":
        return np.sqrt(np.clip(raw_errors, a_min=0.0, a_max=None))
    raise ValueError(f"Unsupported variant: {variant}")


def summarize_variant(variant_errors: np.ndarray, masks: np.ndarray, is_forged: np.ndarray) -> dict[str, object]:
    combined = variant_errors.mean(axis=1)
    image_mean = combined.mean(axis=(1, 2))
    image_p95 = np.percentile(combined.reshape(combined.shape[0], -1), 95, axis=1)
    image_p99 = np.percentile(combined.reshape(combined.shape[0], -1), 99, axis=1)

    authentic = ~is_forged
    forged = is_forged

    forged_fg_mean = []
    forged_bg_mean = []
    forged_fg_bg_gap = []
    forged_fg_bg_ratio = []
    forged_topk_precision = []

    for score_map, mask in zip(combined[forged], masks[forged]):
        fg_values = score_map[mask > 0]
        bg_values = score_map[mask == 0]
        if fg_values.size == 0 or bg_values.size == 0:
            continue
        fg_mean = float(fg_values.mean())
        bg_mean = float(bg_values.mean())
        forged_fg_mean.append(fg_mean)
        forged_bg_mean.append(bg_mean)
        forged_fg_bg_gap.append(fg_mean - bg_mean)
        forged_fg_bg_ratio.append(fg_mean / max(bg_mean, 1e-6))
        forged_topk_precision.append(topk_precision(score_map, mask))

    return {
        "authentic_image_mean": summarize_array(image_mean[authentic]),
        "authentic_image_p95": summarize_array(image_p95[authentic]),
        "authentic_image_p99": summarize_array(image_p99[authentic]),
        "forged_image_mean": summarize_array(image_mean[forged]),
        "forged_image_p95": summarize_array(image_p95[forged]),
        "forged_image_p99": summarize_array(image_p99[forged]),
        "forged_fg_mean": summarize_array(np.asarray(forged_fg_mean, dtype=np.float32)),
        "forged_bg_mean": summarize_array(np.asarray(forged_bg_mean, dtype=np.float32)),
        "forged_fg_bg_gap": summarize_array(np.asarray(forged_fg_bg_gap, dtype=np.float32)),
        "forged_fg_bg_ratio": summarize_array(np.asarray(forged_fg_bg_ratio, dtype=np.float32)),
        "forged_topk_precision": summarize_array(np.asarray(forged_topk_precision, dtype=np.float32)),
        "authentic_vs_forged_p99_auc": roc_auc_score_binary(is_forged.astype(np.int64), image_p99),
    }


def summarize_raw_scales(raw_errors: np.ndarray, masks: np.ndarray, is_forged: np.ndarray) -> list[dict[str, object]]:
    summaries = []
    authentic = ~is_forged
    forged = is_forged

    for scale_idx in range(raw_errors.shape[1]):
        scale_map = raw_errors[:, scale_idx]
        image_p99 = np.percentile(scale_map.reshape(scale_map.shape[0], -1), 99, axis=1)
        forged_fg_mean = []
        forged_bg_mean = []

        for score_map, mask in zip(scale_map[forged], masks[forged]):
            fg_values = score_map[mask > 0]
            bg_values = score_map[mask == 0]
            if fg_values.size == 0 or bg_values.size == 0:
                continue
            forged_fg_mean.append(float(fg_values.mean()))
            forged_bg_mean.append(float(bg_values.mean()))

        summaries.append(
            {
                "scale_index": scale_idx,
                "authentic_image_p99": summarize_array(image_p99[authentic]),
                "forged_image_p99": summarize_array(image_p99[forged]),
                "forged_fg_mean": summarize_array(np.asarray(forged_fg_mean, dtype=np.float32)),
                "forged_bg_mean": summarize_array(np.asarray(forged_bg_mean, dtype=np.float32)),
                "authentic_vs_forged_p99_auc": roc_auc_score_binary(is_forged.astype(np.int64), image_p99),
            }
        )

    return summaries


def summarize_branch_outputs(branch_values: dict[str, list[float]], is_forged: np.ndarray) -> dict[str, object]:
    authentic = ~is_forged
    forged = is_forged
    summary = {}

    for key, values in branch_values.items():
        values_np = np.asarray(values, dtype=np.float32)
        summary[key] = {
            "authentic": summarize_array(values_np[authentic]),
            "forged": summarize_array(values_np[forged]),
        }

    return summary


def save_visual_grid(
    output_path: Path,
    images: np.ndarray,
    masks: np.ndarray,
    raw_errors: np.ndarray,
    variants: dict[str, np.ndarray],
    branch_maps: dict[str, np.ndarray] | None,
    is_forged: np.ndarray,
    image_paths: list[str],
) -> None:
    chosen_indices = []
    authentic_indices = [idx for idx, forged in enumerate(is_forged) if not forged][:2]
    forged_indices = [idx for idx, forged in enumerate(is_forged) if forged][:2]
    chosen_indices.extend(authentic_indices)
    chosen_indices.extend(forged_indices)
    if not chosen_indices:
        return

    columns = [
        "image",
        "gt",
        "raw",
        "log1p",
        "zscore",
    ]
    if branch_maps is not None:
        columns.extend(["dlf", "se", "refined"])

    fig, axes = plt.subplots(len(chosen_indices), len(columns), figsize=(3.2 * len(columns), 3.2 * len(chosen_indices)))
    if len(chosen_indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    raw_combined = raw_errors.mean(axis=1)
    log_combined = variants["log1p"].mean(axis=1)
    zscore_combined = variants["zscore"].mean(axis=1)

    for row_idx, sample_idx in enumerate(chosen_indices):
        image = np.transpose(images[sample_idx], (1, 2, 0))
        mask = masks[sample_idx]
        raw_map = raw_combined[sample_idx]
        log_map = log_combined[sample_idx]
        zscore_map = zscore_combined[sample_idx]

        raw_vmax = max(float(np.percentile(raw_map, 99.0)), 1e-6)
        log_vmax = max(float(np.percentile(log_map, 99.0)), 1e-6)

        panels = [
            (image, image_paths[sample_idx], None, None),
            (mask, "gt", "gray", None),
            (raw_map, "raw", "magma", (0.0, raw_vmax)),
            (log_map, "log1p", "magma", (0.0, log_vmax)),
            (zscore_map, "zscore", "coolwarm", (-3.0, 3.0)),
        ]

        if branch_maps is not None:
            panels.extend(
                [
                    (branch_maps["dlf"][sample_idx], "dlf", "viridis", (0.0, 1.0)),
                    (branch_maps["se"][sample_idx], "se", "viridis", (0.0, 1.0)),
                    (branch_maps["refined"][sample_idx], "refined", "viridis", (0.0, 1.0)),
                ]
            )

        for col_idx, (panel, title, cmap, limits) in enumerate(panels):
            axis = axes[row_idx, col_idx]
            if limits is None:
                axis.imshow(panel)
            else:
                axis.imshow(panel, cmap=cmap, vmin=limits[0], vmax=limits[1])
            axis.set_title(title)
            axis.axis("off")

        label = "forged" if is_forged[sample_idx] else "authentic"
        axes[row_idx, 0].set_ylabel(label, rotation=90, fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diagnose_dlf] Using device: {device}")

    val_dataset = build_validation_dataset(args)
    if val_dataset is None or len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty.")

    selected_indices = select_balanced_indices(val_dataset, samples_per_class=args.samples_per_class)
    if not selected_indices:
        raise RuntimeError("Could not find any validation samples to analyze.")

    analysis_dataset = Subset(val_dataset, selected_indices)
    loader = DataLoader(analysis_dataset, batch_size=args.batch_size, shuffle=False)
    pm_backbone, dino_extractor, pyramid_zm = build_feature_extractors(args, device)

    checkpoint = None
    if args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        print(f"[diagnose_dlf] Loaded checkpoint metadata from {args.checkpoint}")

    dlf_decoder = None
    se_model = None

    raw_errors_list = []
    masks_list = []
    images_list = []
    image_paths = []
    is_forged_list = []
    branch_values = {
        "dlf_positive_rate": [],
        "se_positive_rate": [],
        "refined_positive_rate": [],
        "dlf_components": [],
        "se_components": [],
        "refined_components": [],
    }
    branch_maps = {
        "dlf": [],
        "se": [],
        "refined": [],
    }

    with torch.no_grad():
        for batch_idx, (images, masks, labels, batch_paths) in enumerate(loader, start=1):
            images = images.to(device)
            masks = masks.to(device=device, dtype=torch.float32)

            cnn_raw_errors, zernike_raw_errors, cnn_branch_result, zernike_branch_result, dino_features, _ = extract_localization_inputs(
                images=images,
                pm_backbone=pm_backbone,
                pyramid_zm=pyramid_zm,
                dino_extractor=dino_extractor,
                separate_transforms=args.separate_transforms,
                cnn_feature_norm=args.cnn_feature_norm,
                pm_random_window=args.pm_random_window,
                pm_iters=args.pm_iters,
                pm_beta=args.pm_beta,
                pm_hard_selection=args.pm_hard_selection,
                pm_use_non_local=args.pm_use_non_local,
                pm_non_local_limit=args.pm_non_local_limit,
                pm_flat_threshold=args.pm_flat_threshold,
                pm_margin_threshold=args.pm_margin_threshold,
                pm_topk=args.pm_topk,
                pm_reduced_precision=args.pm_reduced_precision,
                localization_resolution=args.localization_resolution,
                dlf_error_scaling="none",
            )
            raw_errors = torch.cat((cnn_raw_errors, zernike_raw_errors), dim=1)

            if dlf_decoder is None and checkpoint is not None:
                dlf_decoder = DLFDecoder(
                    num_error_maps=cnn_raw_errors.shape[1],
                ).to(device)
                load_module_state(dlf_decoder, checkpoint["dlf_decoder"], "dlf_decoder")
                dlf_decoder.eval()

                se_model = SEUNet(
                    in_channels=dino_features.shape[1],
                    out_channels=1,
                    final_activation="sigmoid",
                ).to(device)
                load_module_state(se_model, checkpoint["se_model"], "se_model")
                se_model.eval()

            raw_errors_list.append(raw_errors.cpu().numpy())
            masks_list.append(masks.cpu().numpy().astype(np.uint8))
            images_list.append(images.cpu().numpy())
            image_paths.extend(batch_paths)
            is_forged_list.extend(bool(mask.any().item()) for mask in masks)

            if dlf_decoder is not None and se_model is not None:
                scaled_cnn_errors = normalize_dlf_error_maps(cnn_raw_errors, mode=args.current_scaling)
                scaled_zernike_errors = normalize_dlf_error_maps(zernike_raw_errors, mode=args.current_scaling)
                refined_mask, target_map, dlf_map = decode_and_refine_masks(
                    images=images,
                    cnn_error_maps=scaled_cnn_errors,
                    zernike_error_maps=scaled_zernike_errors,
                    cnn_branch_result=cnn_branch_result,
                    zernike_branch_result=zernike_branch_result,
                    dlf_decoder=dlf_decoder,
                    se_model=se_model,
                    dino_features=dino_features,
                    output_size=images.shape[-2:],
                )

                dlf_np = dlf_map.squeeze(1).cpu().numpy()
                se_np = target_map.squeeze(1).cpu().numpy()
                refined_np = refined_mask.squeeze(1).cpu().numpy()

                branch_maps["dlf"].append(dlf_np)
                branch_maps["se"].append(se_np)
                branch_maps["refined"].append(refined_np)

                for dlf_sample, se_sample, refined_sample in zip(dlf_np, se_np, refined_np):
                    dlf_mask = dlf_sample >= 0.5
                    se_mask = se_sample >= 0.5
                    refined_binary = refined_sample >= 0.5
                    branch_values["dlf_positive_rate"].append(float(dlf_mask.mean()))
                    branch_values["se_positive_rate"].append(float(se_mask.mean()))
                    branch_values["refined_positive_rate"].append(float(refined_binary.mean()))
                    branch_values["dlf_components"].append(float(count_components(dlf_mask)))
                    branch_values["se_components"].append(float(count_components(se_mask)))
                    branch_values["refined_components"].append(float(count_components(refined_binary)))

            print(f"[diagnose_dlf] Processed batch {batch_idx}/{len(loader)}")

    raw_errors = np.concatenate(raw_errors_list, axis=0)
    masks = np.concatenate(masks_list, axis=0)
    images = np.concatenate(images_list, axis=0)
    is_forged = np.asarray(is_forged_list, dtype=bool)

    variants = {
        "none": transform_variant(raw_errors, "none"),
        "sqrt": transform_variant(raw_errors, "sqrt"),
        "log1p": transform_variant(raw_errors, "log1p"),
        "zscore": transform_variant(raw_errors, "zscore"),
    }

    branch_maps_np = None
    if branch_maps["dlf"]:
        branch_maps_np = {key: np.concatenate(value, axis=0) for key, value in branch_maps.items()}

    summary = {
        "config": {
            "checkpoint": str(args.checkpoint),
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "samples_per_class": args.samples_per_class,
            "validation_split": args.validation_split,
            "validation_seed": args.validation_seed,
            "pm_iters": args.pm_iters,
            "pm_beta": args.pm_beta,
            "pm_hard_selection": args.pm_hard_selection,
            "pm_use_non_local": args.pm_use_non_local,
            "pm_non_local_limit": args.pm_non_local_limit,
            "pm_flat_threshold": args.pm_flat_threshold,
            "pm_margin_threshold": args.pm_margin_threshold,
            "pm_topk": args.pm_topk,
            "localization_resolution": args.localization_resolution,
            "current_scaling": args.current_scaling,
        },
        "sample_counts": {
            "total": int(len(is_forged)),
            "authentic": int((~is_forged).sum()),
            "forged": int(is_forged.sum()),
        },
        "raw_scale_summaries": summarize_raw_scales(raw_errors, masks, is_forged),
        "variant_summaries": {
            name: summarize_variant(variant_errors, masks, is_forged)
            for name, variant_errors in variants.items()
        },
    }

    if branch_maps_np is not None:
        summary["current_branch_summary"] = summarize_branch_outputs(branch_values, is_forged)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    visual_path = args.output_dir / "visual_grid.png"
    save_visual_grid(
        output_path=visual_path,
        images=images,
        masks=masks,
        raw_errors=raw_errors,
        variants=variants,
        branch_maps=branch_maps_np,
        is_forged=is_forged,
        image_paths=image_paths,
    )

    print(f"[diagnose_dlf] Wrote summary to {summary_path}")
    print(f"[diagnose_dlf] Wrote visual grid to {visual_path}")


if __name__ == "__main__":
    main()
