from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, v2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import (
    Datasets,
    normalize_mask_array,
    resolve_data_root,
)
from inference_helpers import (
    predict_binary_mask,
    resolve_inference_transform,
    resolve_runtime_device,
    restore_feature_branches_from_checkpoint,
    restore_localization_heads_from_checkpoint,
    safe_prediction_stem,
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
from training.losses import localization_loss_terms, summarize_branch_activity
from training.metrics_logging import (
    build_validation_summary,
    initialize_instance_metric_tracker,
    initialize_metric_accumulator,
    update_instance_metric_tracker,
    update_metric_accumulator,
)


@dataclass(frozen=True)
class SelectedSampleRecord:
    path: Path
    label: int
    mask_dir: Path | None
    dataset_name: str

    @property
    def is_forged(self) -> bool:
        return "forged" in self.path.parent.name


class SelectedSamplesDataset(Dataset):
    def __init__(
        self,
        records: list[SelectedSampleRecord],
        image_size: int,
        transform,
    ):
        self.records = records
        self.image_transforms = transform(image_size)
        self.mask_transforms = v2.Compose(
            [
                v2.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.NEAREST,
                ),
            ]
        )
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]

        image = Image.open(record.path).convert("RGB")
        image = self.image_transforms(image)

        if record.mask_dir is not None and record.is_forged:
            mask_path = record.mask_dir / record.path.name.replace(".png", ".npy")
            mask = np.load(mask_path)
            mask = normalize_mask_array(mask)
            mask = torch.from_numpy(mask)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = self.mask_transforms(mask)
            mask = mask.squeeze(0).long()
        else:
            mask = torch.zeros((self.image_size, self.image_size), dtype=torch.long)

        return image, mask, str(record.path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a checkpoint on an explicit sample list such as "
            "heldout_test_samples.txt and save a summary plus per-sample metrics."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=ROOT / "artifacts" / "final_ver_run",
        help="Run directory containing checkpoints/best.pt and heldout_test_samples.txt.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint override. Defaults to <run_dir>/checkpoints/best.pt.",
    )
    parser.add_argument(
        "--samples-file",
        type=Path,
        default=None,
        help="Optional sample-list override. Defaults to <run_dir>/heldout_test_samples.txt.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional split-manifest override. Defaults to <run_dir>/split_manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save summary files. Defaults to <run_dir>/heldout_test_eval.",
    )
    parser.add_argument("--image-size", type=int, default=488)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--feature-backbone", choices=("dino", "dino_single"), default="dino_single")
    parser.add_argument("--use-dino-transform", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dino-model-name", default="dinov2_vits14")
    parser.add_argument("--cnn-feature-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--separate-transforms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pm-iters", type=int, default=24)
    parser.add_argument("--pm-beta", type=float, default=10.0)
    parser.add_argument("--pm-hard-selection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pm-random-window", type=int, default=50)
    parser.add_argument("--pm-use-non-local", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pm-non-local-limit", type=float, default=25.0)
    parser.add_argument("--pm-flat-threshold", type=float, default=0.15)
    parser.add_argument("--pm-margin-threshold", type=float, default=0.10)
    parser.add_argument("--pm-topk", type=int, default=1)
    parser.add_argument("--pm-reduced-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--localization-resolution", choices=("image",), default="image")
    parser.add_argument("--dlf-error-scaling", choices=("none", "log1p", "zscore"), default="log1p")
    parser.add_argument("--mprime-loss-weight", type=float, default=0.8)
    parser.add_argument("--empty-target-penalty-weight", type=float, default=0.25)
    parser.add_argument("--post-process-threshold", type=float, default=0.6)
    parser.add_argument("--post-process-confident-threshold", type=float, default=0.9)
    parser.add_argument("--post-process-min-component-area", type=int, default=128)
    parser.add_argument("--post-process-fill-holes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-process-apply-closing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-process-smooth-probabilities", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-post-process", action="store_true")
    parser.add_argument("--raw-threshold", type=float, default=0.5)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def resolve_checkpoint_path(run_dir: Path, checkpoint_override: Path | None) -> Path:
    if checkpoint_override is not None:
        return checkpoint_override.resolve()
    return (run_dir / "checkpoints" / "best.pt").resolve()


def resolve_existing_path(path: Path) -> Path:
    candidates = [path, ROOT / path, ROOT.parent / path, resolve_data_root() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve path: {path}")


def build_manifest_lookup(split_manifest_path: Path | None) -> dict[str, tuple[Path | None, str]]:
    if split_manifest_path is None or not split_manifest_path.exists():
        return {}

    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    lookup: dict[str, tuple[Path | None, str]] = {}
    for dataset_entry in manifest.get("datasets", []):
        mask_dir = dataset_entry.get("mask_dir")
        mask_dir_path = Path(mask_dir) if mask_dir is not None else None
        dataset_name = dataset_entry["dataset_name"]
        for key in ("train_samples", "val_samples", "test_samples"):
            for sample_path in dataset_entry.get(key, []):
                lookup[str(Path(sample_path).resolve())] = (mask_dir_path, dataset_name)
    return lookup


def infer_dataset_metadata(sample_path: Path) -> tuple[Path | None, str]:
    root = resolve_data_root().resolve()
    for dataset in Datasets.ALL_TRAIN.value:
        image_root = (root / dataset["images"]).resolve()
        if sample_path.is_relative_to(image_root):
            mask_dir = root / dataset["masks"] if dataset["masks"] is not None else None
            return mask_dir, Path(dataset["images"]).name
    raise FileNotFoundError(f"Could not map sample {sample_path} to a known dataset root.")


def load_selected_samples(
    samples_file: Path,
    split_manifest_path: Path | None,
    limit: int | None,
) -> list[SelectedSampleRecord]:
    manifest_lookup = build_manifest_lookup(split_manifest_path)
    records: list[SelectedSampleRecord] = []

    lines = [
        line.strip()
        for line in samples_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        lines = lines[:limit]

    for raw_path in lines:
        sample_path = resolve_existing_path(Path(raw_path))
        metadata = manifest_lookup.get(str(sample_path))
        if metadata is None:
            mask_dir, dataset_name = infer_dataset_metadata(sample_path)
        else:
            mask_dir, dataset_name = metadata
        label = 1 if "forged" in sample_path.parent.name else 0
        records.append(
            SelectedSampleRecord(
                path=sample_path,
                label=label,
                mask_dir=mask_dir,
                dataset_name=dataset_name,
            )
        )

    return records


def save_prediction_png(predictions_dir: Path, sample_path: str, pred_mask: np.ndarray) -> None:
    output_path = predictions_dir / f"{safe_prediction_stem(sample_path)}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((pred_mask.astype(np.uint8) * 255)).save(output_path)


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("medium")

    run_dir = args.run_dir.resolve()
    checkpoint_path = resolve_checkpoint_path(run_dir, args.checkpoint)
    samples_file = (args.samples_file or (run_dir / "heldout_test_samples.txt")).resolve()
    split_manifest_path = (args.split_manifest or (run_dir / "split_manifest.json")).resolve()
    output_dir = (args.output_dir or (run_dir / "heldout_test_eval")).resolve()
    predictions_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    records = load_selected_samples(samples_file, split_manifest_path, args.limit)
    if not records:
        raise RuntimeError(f"No samples found in {samples_file}")

    device = resolve_runtime_device(args.device)
    print(f"[heldout-eval] Using device: {device}")
    print(f"[heldout-eval] Checkpoint: {checkpoint_path}")
    print(f"[heldout-eval] Samples file: {samples_file}")
    print(f"[heldout-eval] Loaded {len(records)} samples")

    dataset = SelectedSamplesDataset(
        records=records,
        image_size=args.image_size,
        transform=resolve_inference_transform(
            feature_backbone=args.feature_backbone,
            use_dino_transform=args.use_dino_transform,
            separate_transforms=args.separate_transforms,
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    pm_backbone, dino_extractor, pyramid_zm = restore_feature_branches_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
        feature_backbone=args.feature_backbone,
        dino_model_name=args.dino_model_name,
        separate_transforms=args.separate_transforms,
        use_dino_transform=args.use_dino_transform,
    )

    dlf_decoder = None
    se_model = None
    util = None if args.disable_post_process else MaskUtil()
    mask_dir_by_sample = {str(record.path): record.mask_dir for record in records}
    dataset_name_by_sample = {str(record.path): record.dataset_name for record in records}

    val_accumulator = initialize_metric_accumulator()
    val_loss_steps = 0
    val_counts = initialize_segmentation_counts()
    val_instance_tracker = initialize_instance_metric_tracker()
    per_sample_records: list[dict[str, object]] = []

    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (images, masks, image_paths) in enumerate(loader, start=1):
            batch_start = time.perf_counter()
            images = images.to(device)
            masks = masks.to(device=device, dtype=torch.float32)

            cnn_errors, zernike_errors, cnn_branch_result, zernike_branch_result, dino_features, _ = (
                extract_localization_inputs(
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
                    dlf_error_scaling=args.dlf_error_scaling,
                    collect_stats=False,
                )
            )

            if dlf_decoder is None or se_model is None:
                dlf_decoder, se_model = restore_localization_heads_from_checkpoint(
                    checkpoint,
                    cnn_errors=cnn_errors,
                    dino_features=dino_features,
                    device=device,
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
                mprime_loss_weight=args.mprime_loss_weight,
                empty_target_penalty_weight=args.empty_target_penalty_weight,
            )
            branch_stats = summarize_branch_activity(dlf_map, target_map)
            update_metric_accumulator(
                val_accumulator,
                (
                    val_loss,
                    val_ldfm,
                    val_lmrd,
                    val_mprime_loss,
                    val_empty_target_loss,
                    val_empty_refined_loss,
                    val_empty_target_map_loss,
                    val_empty_mprime_map_loss,
                ),
                branch_stats,
            )
            val_loss_steps += 1

            mask_preds = predict_binary_mask(
                refined_mask,
                disable_post_process=args.disable_post_process,
                raw_threshold=args.raw_threshold,
                post_process_threshold=args.post_process_threshold,
                post_process_confident_threshold=args.post_process_confident_threshold,
                post_process_min_component_area=args.post_process_min_component_area,
                post_process_smooth_probabilities=args.post_process_smooth_probabilities,
                post_process_fill_holes=args.post_process_fill_holes,
                post_process_apply_closing=args.post_process_apply_closing,
                util=util,
            )

            update_segmentation_counts(mask_preds, masks.long(), val_counts)

            refined_probs_np = refined_mask.squeeze(1).cpu().numpy().astype(np.float32)
            target_probs_np = target_map.squeeze(1).cpu().numpy().astype(np.float32)
            dlf_probs_np = dlf_map.squeeze(1).cpu().numpy().astype(np.float32)
            pred_masks_np = mask_preds.cpu().numpy().astype(np.uint8)

            for sample_idx, image_path in enumerate(image_paths):
                pred_mask = pred_masks_np[sample_idx]
                pred_instances = binary_mask_to_instances(pred_mask)
                gt_instances = load_resized_gt_instances(
                    image_path,
                    mask_dir_by_sample=mask_dir_by_sample,
                    image_size=args.image_size,
                )
                image_of1 = optimal_f1_score(pred_instances, gt_instances)
                pred_component_count = len(pred_instances)
                gt_component_count = len(gt_instances)
                is_forged = gt_component_count > 0

                update_instance_metric_tracker(
                    val_instance_tracker,
                    image_of1=image_of1,
                    pred_component_count=pred_component_count,
                    gt_component_count=gt_component_count,
                )

                per_sample_records.append(
                    {
                        "image_path": image_path,
                        "dataset_name": dataset_name_by_sample[image_path],
                        "is_forged": is_forged,
                        "image_of1": float(image_of1),
                        "pred_component_count": pred_component_count,
                        "gt_component_count": gt_component_count,
                        "pred_positive_rate": float(pred_mask.mean()),
                        "mean_refined_probability": float(refined_probs_np[sample_idx].mean()),
                        "max_refined_probability": float(refined_probs_np[sample_idx].max()),
                        "mean_target_probability": float(target_probs_np[sample_idx].mean()),
                        "mean_dlf_probability": float(dlf_probs_np[sample_idx].mean()),
                    }
                )

                if args.save_predictions:
                    save_prediction_png(predictions_dir, image_path, pred_mask)

            if args.log_every > 0 and (batch_idx % args.log_every == 0 or batch_idx == len(loader)):
                processed = min(batch_idx * args.batch_size, len(records))
                avg_of1 = val_instance_tracker["of1_sum"] / max(int(val_instance_tracker["images"]), 1)
                print(
                    f"[heldout-eval] batch {batch_idx}/{len(loader)} "
                    f"processed={processed}/{len(records)} "
                    f"avg_oF1={avg_of1:.4f} "
                    f"elapsed={time.perf_counter() - batch_start:.2f}s"
                )

    total_seconds = time.perf_counter() - start_time

    val_metrics = summarize_segmentation_counts(val_counts)
    val_summary = build_validation_summary(
        val_accumulator,
        val_loss_steps,
        val_metrics,
        val_instance_tracker,
    )

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "run_dir": str(run_dir),
        "samples_file": str(samples_file),
        "split_manifest": str(split_manifest_path) if split_manifest_path.exists() else None,
        "sample_counts": {
            "total": len(records),
            "authentic": int(val_instance_tracker["authentic_images"]),
            "forged": int(val_instance_tracker["forged_images"]),
        },
        "config": {
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "feature_backbone": args.feature_backbone,
            "dino_model_name": args.dino_model_name,
            "pm_iters": args.pm_iters,
            "pm_beta": args.pm_beta,
            "pm_hard_selection": args.pm_hard_selection,
            "pm_random_window": args.pm_random_window,
            "pm_use_non_local": args.pm_use_non_local,
            "pm_non_local_limit": args.pm_non_local_limit,
            "pm_flat_threshold": args.pm_flat_threshold,
            "pm_margin_threshold": args.pm_margin_threshold,
            "pm_topk": args.pm_topk,
            "pm_reduced_precision": args.pm_reduced_precision,
            "dlf_error_scaling": args.dlf_error_scaling,
            "mprime_loss_weight": args.mprime_loss_weight,
            "empty_target_penalty_weight": args.empty_target_penalty_weight,
            "post_process_enabled": not args.disable_post_process,
            "post_process_threshold": args.post_process_threshold,
            "post_process_confident_threshold": args.post_process_confident_threshold,
            "post_process_min_component_area": args.post_process_min_component_area,
            "post_process_fill_holes": args.post_process_fill_holes,
            "post_process_apply_closing": args.post_process_apply_closing,
            "post_process_smooth_probabilities": args.post_process_smooth_probabilities,
            "raw_threshold": args.raw_threshold if args.disable_post_process else None,
            "save_predictions": args.save_predictions,
        },
        "metrics": {
            "val_loss": val_summary["loss"],
            "val_ldfm": val_summary["ldfm"],
            "val_lmrd": val_summary["lmrd"],
            "val_mprime_loss": val_summary["mprime_loss"],
            "val_empty_target_loss": val_summary["empty_target_loss"],
            "val_empty_refined_loss": val_summary["empty_refined_loss"],
            "val_empty_target_map_loss": val_summary["empty_target_map_loss"],
            "val_empty_mprime_map_loss": val_summary["empty_mprime_map_loss"],
            "val_mprime_positive_rate": val_summary["mprime_positive_rate"],
            "val_mprime_wins_rate": val_summary["mprime_wins_rate"],
            "val_target_positive_rate": val_summary["target_positive_rate"],
            "val_of1": val_summary["of1"],
            "val_pred_components_per_image": val_summary["pred_components_per_image"],
            "val_authentic_of1": val_summary["authentic_of1"],
            "val_authentic_empty_pred_rate": val_summary["authentic_empty_pred_rate"],
            "val_authentic_pred_components_per_image": val_summary["authentic_pred_components_per_image"],
            "val_forged_of1": val_summary["forged_of1"],
            "val_forged_pred_components_per_image": val_summary["forged_pred_components_per_image"],
            "val_forged_gt_components_per_image": val_summary["forged_gt_components_per_image"],
            "val_iou": val_summary["iou"],
            "val_dice": val_summary["dice"],
            "val_pred_positive_rate": val_summary["pred_positive_rate"],
            "val_mask_positive_rate": val_summary["mask_positive_rate"],
        },
        "runtime": {
            "total_seconds": total_seconds,
            "seconds_per_image": total_seconds / max(len(records), 1),
        },
    }

    summary_path = output_dir / "summary.json"
    per_sample_path = output_dir / "per_sample_metrics.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with per_sample_path.open("w", encoding="utf-8") as handle:
        for row in per_sample_records:
            handle.write(json.dumps(row) + "\n")

    print(
        "[heldout-eval] "
        f"val_loss={val_summary['loss']:.4f} "
        f"val_oF1={val_summary['of1']:.4f} "
        f"auth_oF1={val_summary['authentic_of1']:.4f} "
        f"forged_oF1={val_summary['forged_of1']:.4f} "
        f"pred_pos={val_summary['pred_positive_rate']:.4%}"
    )
    print(f"[heldout-eval] summary={summary_path}")
    print(f"[heldout-eval] per_sample={per_sample_path}")
    if args.save_predictions:
        print(f"[heldout-eval] predictions={predictions_dir}")


if __name__ == "__main__":
    main()
