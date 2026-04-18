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
    resolve_image_transform,
)
from feature_extractors.cnn_feature_extractor import (
    PretrainedBackboneExtractor,
    SingleScaleFeatureExtractor,
)
from feature_extractors.dino_feature_extractor import (
    PyramidDinoFeatureExtractor,
    SingleScaleDinoFeatureExtractor,
)
from feature_extractors.zernike_feature_extractor import (
    PyramidZernikeExtractor,
    default_pq_list,
)
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
from training.checkpointing import load_module_state
from training.losses import localization_loss_terms, summarize_branch_activity


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
    parser.add_argument("--post-process-min-component-area", type=int, default=32)
    parser.add_argument("--post-process-fill-holes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-process-apply-closing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-process-smooth-probabilities", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-post-process", action="store_true")
    parser.add_argument("--raw-threshold", type=float, default=0.5)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def choose_transform(args: argparse.Namespace):
    return resolve_image_transform(
        feature_backbone=args.feature_backbone,
        use_dino_transform=args.use_dino_transform,
        cnn_backbone="pretrained",
        separate_transforms=args.separate_transforms,
    )


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def build_feature_extractors(args: argparse.Namespace, device: torch.device):
    dino_extractor_cls = (
        PyramidDinoFeatureExtractor if args.feature_backbone == "dino" else SingleScaleDinoFeatureExtractor
    )
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


def ensure_module_loaded(module, state_dict: dict[str, torch.Tensor], module_name: str) -> None:
    if not load_module_state(module, state_dict, module_name):
        raise RuntimeError(f"Failed to fully restore {module_name} from checkpoint.")


def build_models_from_checkpoint(
    checkpoint: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
):
    pm_backbone, dino_extractor, pyramid_zm = build_feature_extractors(args, device)
    ensure_module_loaded(pm_backbone, checkpoint["pm_backbone"], "pm_backbone")
    if "dino_extractor" in checkpoint:
        ensure_module_loaded(dino_extractor, checkpoint["dino_extractor"], "dino_extractor")
    elif "pyramid_bb" in checkpoint:
        ensure_module_loaded(dino_extractor, checkpoint["pyramid_bb"], "dino_extractor")
    else:
        raise KeyError("Checkpoint is missing dino_extractor/pyramid_bb weights.")

    return pm_backbone, dino_extractor, pyramid_zm


def safe_prediction_stem(sample_path: str) -> str:
    path = Path(sample_path)
    data_root = resolve_data_root().resolve()
    try:
        relative = path.resolve().relative_to(data_root)
        return "__".join(relative.with_suffix("").parts)
    except ValueError:
        return path.stem


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

    device = resolve_device(args)
    print(f"[heldout-eval] Using device: {device}")
    print(f"[heldout-eval] Checkpoint: {checkpoint_path}")
    print(f"[heldout-eval] Samples file: {samples_file}")
    print(f"[heldout-eval] Loaded {len(records)} samples")

    dataset = SelectedSamplesDataset(
        records=records,
        image_size=args.image_size,
        transform=choose_transform(args),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    pm_backbone, dino_extractor, pyramid_zm = build_models_from_checkpoint(
        checkpoint=checkpoint,
        args=args,
        device=device,
    )

    dlf_decoder = None
    se_model = None
    util = None if args.disable_post_process else MaskUtil()
    mask_dir_by_sample = {str(record.path): record.mask_dir for record in records}
    dataset_name_by_sample = {str(record.path): record.dataset_name for record in records}

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
                dlf_decoder = DLFDecoder(
                    num_error_maps=cnn_errors.shape[1],
                ).to(device)
                ensure_module_loaded(dlf_decoder, checkpoint["dlf_decoder"], "dlf_decoder")
                dlf_decoder.eval()

                se_model = SEUNet(
                    in_channels=dino_features.shape[1],
                    out_channels=1,
                    final_activation="sigmoid",
                ).to(device)
                ensure_module_loaded(se_model, checkpoint["se_model"], "se_model")
                se_model.eval()

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
            if args.disable_post_process:
                mask_preds = (mask_probs >= args.raw_threshold).long()
            else:
                mask_preds = post_process_mask_batch(
                    mask_probs,
                    util,
                    threshold=args.post_process_threshold,
                    confident_threshold=args.post_process_confident_threshold,
                    min_component_area=args.post_process_min_component_area,
                    smooth_probabilities=args.post_process_smooth_probabilities,
                    fill_holes=args.post_process_fill_holes,
                    apply_closing=args.post_process_apply_closing,
                )

            update_segmentation_counts(mask_preds, masks.long(), val_counts)

            refined_probs_np = mask_probs.cpu().numpy().astype(np.float32)
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

                val_of1_sum += image_of1
                val_images += 1
                val_pred_components_sum += pred_component_count

                if is_forged:
                    val_forged_of1_sum += image_of1
                    val_forged_images += 1
                    val_forged_pred_components_sum += pred_component_count
                    val_forged_gt_components_sum += gt_component_count
                else:
                    val_authentic_of1_sum += image_of1
                    val_authentic_images += 1
                    val_authentic_pred_components_sum += pred_component_count
                    if pred_component_count == 0:
                        val_authentic_empty_pred_count += 1

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
                avg_of1 = val_of1_sum / max(val_images, 1)
                print(
                    f"[heldout-eval] batch {batch_idx}/{len(loader)} "
                    f"processed={processed}/{len(records)} "
                    f"avg_oF1={avg_of1:.4f} "
                    f"elapsed={time.perf_counter() - batch_start:.2f}s"
                )

    total_seconds = time.perf_counter() - start_time

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

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "run_dir": str(run_dir),
        "samples_file": str(samples_file),
        "split_manifest": str(split_manifest_path) if split_manifest_path.exists() else None,
        "sample_counts": {
            "total": len(records),
            "authentic": val_authentic_images,
            "forged": val_forged_images,
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
            "val_loss": val_mean_loss,
            "val_ldfm": val_mean_ldfm,
            "val_lmrd": val_mean_lmrd,
            "val_mprime_loss": val_mean_mprime_loss,
            "val_empty_target_loss": val_mean_empty_target_loss,
            "val_empty_refined_loss": val_mean_empty_refined_loss,
            "val_empty_target_map_loss": val_mean_empty_target_map_loss,
            "val_empty_mprime_map_loss": val_mean_empty_mprime_map_loss,
            "val_mprime_positive_rate": val_mean_mprime_positive_rate,
            "val_mprime_wins_rate": val_mean_mprime_wins_rate,
            "val_target_positive_rate": val_mean_target_positive_rate,
            "val_of1": val_mean_of1,
            "val_pred_components_per_image": val_mean_pred_components,
            "val_authentic_of1": val_mean_authentic_of1,
            "val_authentic_empty_pred_rate": val_authentic_empty_pred_rate,
            "val_authentic_pred_components_per_image": val_mean_authentic_pred_components,
            "val_forged_of1": val_mean_forged_of1,
            "val_forged_pred_components_per_image": val_mean_forged_pred_components,
            "val_forged_gt_components_per_image": val_mean_forged_gt_components,
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "val_pred_positive_rate": val_metrics["pred_positive_rate"],
            "val_mask_positive_rate": val_metrics["mask_positive_rate"],
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
        f"val_loss={val_mean_loss:.4f} "
        f"val_oF1={val_mean_of1:.4f} "
        f"auth_oF1={val_mean_authentic_of1:.4f} "
        f"forged_oF1={val_mean_forged_of1:.4f} "
        f"pred_pos={val_metrics['pred_positive_rate']:.4%}"
    )
    print(f"[heldout-eval] summary={summary_path}")
    print(f"[heldout-eval] per_sample={per_sample_path}")
    if args.save_predictions:
        print(f"[heldout-eval] predictions={predictions_dir}")


if __name__ == "__main__":
    main()
