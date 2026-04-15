from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.optimize
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import Datasets, ForgeryDataset, resolve_data_root, resolve_image_transform, split_indices_by_label
from feature_extractors.cnn_feature_extractor import PretrainedBackboneExtractor, SingleScaleFeatureExtractor
from feature_extractors.dino_feature_extractor import PyramidDinoFeatureExtractor, SingleScaleDinoFeatureExtractor
from feature_extractors.zernike_feature_extractor import PyramidZernikeExtractor, default_pq_list
from prediction.decoder import DLFDecoder
from prediction.localization import decode_and_refine_masks, extract_localization_inputs
from prediction.mask_metrics import (
    binary_mask_to_instances,
    calculate_binary_f1,
    load_resized_gt_instances,
    optimal_f1_score,
)
from prediction.pixelmaputil_mask import MaskUtil
from prediction.pixelmaputil_mask import post_process_mask_batch
from prediction.se_u_net import SEUNet
from training.checkpointing import load_module_state


@dataclass
class SampleRecord:
    path: Path
    label: int
    mask_dir: Path | None
    dataset_name: str

    @property
    def is_forged(self) -> bool:
        return "forged" in self.path.parent.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single-image blob analysis using a run folder's best checkpoint. "
            "The script saves a visual overlay and a JSON report with per-component confidence and GT matching."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory containing checkpoints/best.pt, or the checkpoint file itself.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save the JSON report and figure. Defaults to <run_dir>/blob_analysis.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional image path. If omitted, the script picks from the validation split.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index within the filtered validation split when --image is not provided.",
    )
    parser.add_argument(
        "--forged-only",
        action="store_true",
        help="When auto-selecting a validation sample, restrict to forged images.",
    )
    parser.add_argument(
        "--authentic-only",
        action="store_true",
        help="When auto-selecting a validation sample, restrict to authentic images.",
    )
    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="List filtered validation candidates and exit.",
    )
    parser.add_argument("--image-size", type=int, default=448)
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
    parser.add_argument("--dino-match-native-resolution", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--localization-resolution", choices=("image", "feature_grid"), default="image")
    parser.add_argument("--dlf-error-scaling", choices=("none", "log1p", "zscore"), default="log1p")
    parser.add_argument("--post-process-threshold", type=float, default=0.6)
    parser.add_argument("--post-process-confident-threshold", type=float, default=0.9)
    parser.add_argument("--post-process-min-component-area", type=int, default=256)
    parser.add_argument("--post-process-fill-holes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-process-apply-closing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-process-smooth-probabilities", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-post-process", action="store_true", help="Inspect raw thresholded predictions instead.")
    parser.add_argument("--raw-threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--show", action="store_true", help="Display the saved figure after writing it.")
    return parser.parse_args()


def choose_transform(args: argparse.Namespace):
    return resolve_image_transform(
        feature_backbone=args.feature_backbone,
        use_dino_transform=args.use_dino_transform,
        cnn_backbone=args.cnn_backbone,
        separate_transforms=args.separate_transforms,
    )


def resolve_checkpoint_path(run_dir: Path) -> tuple[Path, Path]:
    run_dir = run_dir.resolve()
    if run_dir.is_file():
        return run_dir, run_dir.parent.parent if run_dir.parent.name == "checkpoints" else run_dir.parent

    candidate = run_dir / "checkpoints" / "best.pt"
    if candidate.exists():
        return candidate, run_dir

    candidate = run_dir / "best.pt"
    if candidate.exists():
        return candidate, run_dir

    raise FileNotFoundError(f"Could not find best checkpoint under {run_dir}")


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_sample_records(args: argparse.Namespace, validation_only: bool) -> list[SampleRecord]:
    root = resolve_data_root()
    records: list[SampleRecord] = []

    for dataset_idx, dataset in enumerate(Datasets.ALL_TRAIN.value):
        image_folder = ImageFolder(root / dataset["images"])
        samples = [(Path(path), label) for path, label in image_folder.samples]
        mask_dir = root / dataset["masks"] if dataset["masks"] is not None else None

        indices = list(range(len(samples)))
        if validation_only:
            _, indices = split_indices_by_label(
                samples,
                validation_split=args.validation_split,
                seed=args.validation_seed + dataset_idx,
            )

        for idx in indices:
            path, label = samples[idx]
            records.append(
                SampleRecord(
                    path=path,
                    label=label,
                    mask_dir=mask_dir,
                    dataset_name=Path(dataset["images"]).name,
                )
            )
    return records


def resolve_image_path(image_arg: str) -> Path:
    candidates = [
        Path(image_arg),
        ROOT / image_arg,
        resolve_data_root() / image_arg,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve image path: {image_arg}")


def select_sample(args: argparse.Namespace) -> SampleRecord:
    if args.forged_only and args.authentic_only:
        raise ValueError("Choose at most one of --forged-only or --authentic-only.")

    val_records = build_sample_records(args, validation_only=True)

    def matches_filter(record: SampleRecord) -> bool:
        if args.forged_only:
            return record.is_forged
        if args.authentic_only:
            return not record.is_forged
        return True

    filtered = [record for record in val_records if matches_filter(record)]
    if args.list_samples:
        try:
            for idx, record in enumerate(filtered):
                label = "forged" if record.is_forged else "authentic"
                print(f"[{idx}] {label:<9} {record.path}")
        except BrokenPipeError:
            pass
        raise SystemExit(0)

    if args.image is not None:
        target_path = resolve_image_path(args.image)
        all_records = build_sample_records(args, validation_only=False)
        for record in all_records:
            if record.path.resolve() == target_path:
                return record
        raise FileNotFoundError(
            f"Found image {target_path}, but could not map it to a supervised ALL_TRAIN sample."
        )

    if not filtered:
        raise RuntimeError("No samples matched the requested filter.")
    if args.sample_index < 0 or args.sample_index >= len(filtered):
        raise IndexError(f"--sample-index {args.sample_index} is out of range for {len(filtered)} candidates.")
    return filtered[args.sample_index]


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


def load_single_sample(record: SampleRecord, args: argparse.Namespace):
    dataset = ForgeryDataset(
        samples=[(record.path, record.label)],
        mask_dir=record.mask_dir,
        size=args.image_size,
        transform=choose_transform(args),
        return_path=True,
    )
    image, mask, label, image_path = dataset[0]
    return image.unsqueeze(0), mask.unsqueeze(0), torch.tensor([label]), image_path


def load_display_image(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def component_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def component_centroid(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return [0.0, 0.0]
    return [float(xs.mean()), float(ys.mean())]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    inter = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    return float(inter / union) if union > 0 else 0.0


def build_assignment(pred_instances: list[np.ndarray], gt_instances: list[np.ndarray]) -> tuple[np.ndarray, list[int | None]]:
    if not pred_instances or not gt_instances:
        return np.zeros((len(pred_instances), len(gt_instances)), dtype=np.float32), [None] * len(pred_instances)

    f1_matrix = np.zeros((len(pred_instances), len(gt_instances)), dtype=np.float32)
    for pred_idx, pred_mask in enumerate(pred_instances):
        for gt_idx, gt_mask in enumerate(gt_instances):
            f1_matrix[pred_idx, gt_idx] = calculate_binary_f1(pred_mask, gt_mask)

    padded = f1_matrix
    if padded.shape[0] < padded.shape[1]:
        pad_rows = padded.shape[1] - padded.shape[0]
        padded = np.vstack((padded, np.zeros((pad_rows, padded.shape[1]), dtype=np.float32)))

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-padded)
    assigned_gt: list[int | None] = [None] * len(pred_instances)
    for row, col in zip(row_ind.tolist(), col_ind.tolist()):
        if row < len(pred_instances) and col < len(gt_instances):
            assigned_gt[row] = col
    return f1_matrix, assigned_gt


def describe_components(
    pred_instances: list[np.ndarray],
    gt_instances: list[np.ndarray],
    refined_probs: np.ndarray,
    se_probs: np.ndarray,
    dlf_probs: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    f1_matrix, assigned_gt = build_assignment(pred_instances, gt_instances)
    pred_descriptions: list[dict[str, object]] = []
    gt_descriptions: list[dict[str, object]] = []

    for pred_idx, pred_mask in enumerate(pred_instances):
        area = int(pred_mask.sum())
        bbox = component_bbox(pred_mask)
        centroid = component_centroid(pred_mask)
        refined_values = refined_probs[pred_mask > 0]
        se_values = se_probs[pred_mask > 0]
        dlf_values = dlf_probs[pred_mask > 0]

        best_gt_idx = None
        best_gt_f1 = 0.0
        best_gt_iou = 0.0
        if gt_instances:
            best_gt_idx = int(np.argmax(f1_matrix[pred_idx]))
            best_gt_f1 = float(f1_matrix[pred_idx, best_gt_idx])
            best_gt_iou = mask_iou(pred_mask, gt_instances[best_gt_idx])

        assigned_idx = assigned_gt[pred_idx]
        assigned_f1 = float(f1_matrix[pred_idx, assigned_idx]) if assigned_idx is not None else 0.0
        assigned_iou = mask_iou(pred_mask, gt_instances[assigned_idx]) if assigned_idx is not None else 0.0

        pred_descriptions.append(
            {
                "pred_index": pred_idx,
                "area": area,
                "bbox_xyxy": bbox,
                "centroid_xy": centroid,
                "refined_mean_confidence": float(refined_values.mean()) if refined_values.size else 0.0,
                "refined_max_confidence": float(refined_values.max()) if refined_values.size else 0.0,
                "refined_confidence_mass": float(refined_values.sum()) if refined_values.size else 0.0,
                "confident_pixels_090": int((refined_values >= 0.9).sum()) if refined_values.size else 0,
                "se_mean_confidence": float(se_values.mean()) if se_values.size else 0.0,
                "se_max_confidence": float(se_values.max()) if se_values.size else 0.0,
                "dlf_mean_confidence": float(dlf_values.mean()) if dlf_values.size else 0.0,
                "dlf_max_confidence": float(dlf_values.max()) if dlf_values.size else 0.0,
                "best_gt_index": best_gt_idx,
                "best_gt_f1": best_gt_f1,
                "best_gt_iou": best_gt_iou,
                "assigned_gt_index": assigned_idx,
                "assigned_gt_f1": assigned_f1,
                "assigned_gt_iou": assigned_iou,
            }
        )

    for gt_idx, gt_mask in enumerate(gt_instances):
        best_pred_idx = None
        best_pred_f1 = 0.0
        best_pred_iou = 0.0
        if pred_instances:
            scores = [calculate_binary_f1(pred_mask, gt_mask) for pred_mask in pred_instances]
            best_pred_idx = int(np.argmax(scores))
            best_pred_f1 = float(scores[best_pred_idx])
            best_pred_iou = mask_iou(pred_instances[best_pred_idx], gt_mask)

        gt_descriptions.append(
            {
                "gt_index": gt_idx,
                "area": int(gt_mask.sum()),
                "bbox_xyxy": component_bbox(gt_mask),
                "centroid_xy": component_centroid(gt_mask),
                "best_pred_index": best_pred_idx,
                "best_pred_f1": best_pred_f1,
                "best_pred_iou": best_pred_iou,
            }
        )

    return pred_descriptions, gt_descriptions


def render_component_overlay(
    image: np.ndarray,
    refined_probs: np.ndarray,
    se_probs: np.ndarray,
    dlf_probs: np.ndarray,
    pred_mask: np.ndarray,
    pred_components: list[dict[str, object]],
    gt_instances: list[np.ndarray],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.reshape(-1)

    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(refined_probs, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("Refined Probabilities")
    axes[1].axis("off")

    axes[2].imshow(se_probs, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("SE Probabilities")
    axes[2].axis("off")

    axes[3].imshow(dlf_probs, cmap="magma", vmin=0.0, vmax=1.0)
    axes[3].set_title("DLF Probabilities")
    axes[3].axis("off")

    axes[4].imshow(pred_mask, cmap="gray", vmin=0.0, vmax=1.0)
    axes[4].set_title("Post-Processed Prediction")
    axes[4].axis("off")

    overlay_ax = axes[5]
    overlay_ax.imshow(image)
    overlay_ax.set_title("Prediction vs Ground Truth")
    overlay_ax.axis("off")

    pred_labeled, _ = scipy.ndimage.label(pred_mask.astype(bool))
    for gt_mask in gt_instances:
        overlay_ax.contour(gt_mask.astype(float), levels=[0.5], colors="lime", linewidths=2.0)

    for component in pred_components:
        pred_idx = int(component["pred_index"])
        component_mask = pred_labeled == (pred_idx + 1)
        if component_mask.any():
            overlay_ax.contour(component_mask.astype(float), levels=[0.5], colors="red", linewidths=1.5)
        x0, y0, x1, y1 = component["bbox_xyxy"]
        width = max(x1 - x0, 1)
        height = max(y1 - y0, 1)
        overlay_ax.add_patch(Rectangle((x0, y0), width, height, fill=False, edgecolor="yellow", linewidth=1.0))
        cx, cy = component["centroid_xy"]
        overlay_ax.text(
            cx,
            cy,
            f"#{pred_idx}\n{component['refined_mean_confidence']:.2f}/{component['refined_max_confidence']:.2f}",
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 1.5},
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_component_summary(report: dict[str, object]) -> None:
    print(f"[blob-analysis] image={report['image_path']}")
    print(f"[blob-analysis] checkpoint={report['checkpoint_path']}")
    print(
        "[blob-analysis] "
        f"is_forged={report['is_forged']} "
        f"pred_components={report['pred_component_count']} "
        f"gt_components={report['gt_component_count']} "
        f"image_oF1={report['image_of1']:.4f}"
    )
    print(
        "[blob-analysis] "
        f"pred_positive_rate={report['pred_positive_rate']:.4%} "
        f"mean_refined_prob={report['mean_refined_probability']:.4f} "
        f"max_refined_prob={report['max_refined_probability']:.4f}"
    )
    for component in report["pred_components"]:
        assigned = component["assigned_gt_index"]
        assigned_text = f"gt={assigned}" if assigned is not None else "gt=None"
        print(
            "[blob-analysis] "
            f"pred#{component['pred_index']} "
            f"area={component['area']} "
            f"ref_mean={component['refined_mean_confidence']:.3f} "
            f"ref_max={component['refined_max_confidence']:.3f} "
            f"se_mean={component['se_mean_confidence']:.3f} "
            f"dlf_mean={component['dlf_mean_confidence']:.3f} "
            f"{assigned_text} "
            f"assigned_f1={component['assigned_gt_f1']:.3f} "
            f"best_f1={component['best_gt_f1']:.3f} "
            f"best_iou={component['best_gt_iou']:.3f}"
        )


def main() -> None:
    args = parse_args()
    checkpoint_path, run_dir = resolve_checkpoint_path(args.run_dir)
    output_dir = args.output_dir.resolve() if args.output_dir is not None else (run_dir / "blob_analysis")
    device = resolve_device(args)

    sample = select_sample(args)
    images, masks, labels, image_path = load_single_sample(sample, args)
    display_image = load_display_image(sample.path, args.image_size)
    images = images.to(device)
    masks = masks.to(device=device, dtype=torch.float32)
    labels = labels.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    pm_backbone, dino_extractor, pyramid_zm = build_feature_extractors(args, device)
    if "pm_backbone" in checkpoint:
        load_module_state(pm_backbone, checkpoint["pm_backbone"], "pm_backbone")
    if "dino_extractor" in checkpoint:
        load_module_state(dino_extractor, checkpoint["dino_extractor"], "dino_extractor")
    elif "pyramid_bb" in checkpoint:
        load_module_state(dino_extractor, checkpoint["pyramid_bb"], "dino_extractor")

    with torch.no_grad():
        cnn_errors, zernike_errors, cnn_branch_result, zernike_branch_result, dino_features, _ = extract_localization_inputs(
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

        dlf_decoder = DLFDecoder(
            num_error_maps=cnn_errors.shape[1],
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

        refined_probs = refined_mask.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        se_probs = target_map.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        dlf_probs = dlf_map.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

        if args.disable_post_process:
            pred_mask = (refined_probs >= args.raw_threshold).astype(np.uint8)
        else:
            util = MaskUtil()
            pred_mask = post_process_mask_batch(
                refined_mask.squeeze(1),
                util,
                threshold=args.post_process_threshold,
                confident_threshold=args.post_process_confident_threshold,
                min_component_area=args.post_process_min_component_area,
                smooth_probabilities=args.post_process_smooth_probabilities,
                fill_holes=args.post_process_fill_holes,
                apply_closing=args.post_process_apply_closing,
            )[0].cpu().numpy().astype(np.uint8)

    mask_dir_by_sample = {image_path: sample.mask_dir}
    gt_instances = load_resized_gt_instances(
        image_path,
        mask_dir_by_sample=mask_dir_by_sample,
        image_size=args.image_size,
    )
    pred_instances = binary_mask_to_instances(pred_mask)
    pred_components, gt_components = describe_components(
        pred_instances=pred_instances,
        gt_instances=gt_instances,
        refined_probs=refined_probs,
        se_probs=se_probs,
        dlf_probs=dlf_probs,
    )

    report = {
        "checkpoint_path": str(checkpoint_path),
        "run_dir": str(run_dir),
        "image_path": image_path,
        "dataset_name": sample.dataset_name,
        "is_forged": sample.is_forged,
        "label": int(sample.label),
        "image_of1": float(optimal_f1_score(pred_instances, gt_instances)),
        "pred_component_count": len(pred_instances),
        "gt_component_count": len(gt_instances),
        "pred_positive_rate": float(pred_mask.mean()),
        "mean_refined_probability": float(refined_probs.mean()),
        "max_refined_probability": float(refined_probs.max()),
        "mean_se_probability": float(se_probs.mean()),
        "max_se_probability": float(se_probs.max()),
        "mean_dlf_probability": float(dlf_probs.mean()),
        "max_dlf_probability": float(dlf_probs.max()),
        "post_processing": {
            "enabled": not args.disable_post_process,
            "threshold": args.post_process_threshold,
            "confident_threshold": args.post_process_confident_threshold,
            "min_component_area": args.post_process_min_component_area,
            "fill_holes": args.post_process_fill_holes,
            "apply_closing": args.post_process_apply_closing,
            "smooth_probabilities": args.post_process_smooth_probabilities,
            "raw_threshold": args.raw_threshold if args.disable_post_process else None,
        },
        "pred_components": pred_components,
        "gt_components": gt_components,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    image_stem = Path(image_path).stem
    report_path = output_dir / f"{image_stem}_blob_report.json"
    figure_path = output_dir / f"{image_stem}_blob_overlay.png"

    render_component_overlay(
        image=display_image,
        refined_probs=refined_probs,
        se_probs=se_probs,
        dlf_probs=dlf_probs,
        pred_mask=pred_mask,
        pred_components=pred_components,
        gt_instances=gt_instances,
        output_path=figure_path,
    )

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_component_summary(report)
    print(f"[blob-analysis] report={report_path}")
    print(f"[blob-analysis] figure={figure_path}")

    if args.show:
        image = plt.imread(figure_path)
        plt.figure(figsize=(14, 10))
        plt.imshow(image)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
