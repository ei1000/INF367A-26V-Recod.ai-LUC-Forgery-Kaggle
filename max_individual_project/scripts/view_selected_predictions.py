from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import Datasets, normalize_mask_array, resolve_data_root
from inference_helpers import load_display_image, safe_prediction_stem


@dataclass(frozen=True)
class PredictionRecord:
    image_path: Path
    dataset_name: str
    is_forged: bool
    image_of1: float
    pred_component_count: int
    gt_component_count: int
    pred_positive_rate: float
    mean_refined_probability: float
    prediction_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize saved PNG masks from evaluate_selected_samples.py --save-predictions "
            "together with the original image and optional ground truth."
        )
    )
    parser.add_argument(
        "eval_dir",
        nargs="?",
        type=Path,
        default=ROOT / "artifacts" / "final_ver_run" / "heldout_test_eval",
        help="Held-out evaluation directory containing per_sample_metrics.jsonl and predictions/.",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="Optional per-sample metrics override. Defaults to <eval_dir>/per_sample_metrics.jsonl.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Optional predictions override. Defaults to <eval_dir>/predictions.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Optional summary override. Defaults to <eval_dir>/summary.json.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional absolute or repo-relative source image path to visualize.",
    )
    parser.add_argument("--index", type=int, default=0, help="Start index inside the filtered list.")
    parser.add_argument("--count", type=int, default=1, help="How many filtered samples to render.")
    parser.add_argument("--all", action="store_true", help="Render every filtered sample.")
    parser.add_argument("--list", action="store_true", help="List filtered samples and exit.")
    parser.add_argument("--forged-only", action="store_true")
    parser.add_argument("--authentic-only", action="store_true")
    parser.add_argument(
        "--sort-by",
        choices=("input", "worst-of1", "best-of1", "most-components", "highest-pred-pos"),
        default="worst-of1",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional image path to save the rendered figure.",
    )
    parser.add_argument("--no-show", action="store_true", help="Skip showing the figure interactively.")
    return parser.parse_args()


def resolve_optional_path(path: Path | None, fallback: Path) -> Path:
    if path is None:
        return fallback.resolve()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    root_candidate = (ROOT / path).resolve()
    if root_candidate.exists():
        return root_candidate
    repo_candidate = (ROOT.parent / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return cwd_candidate


def resolve_image_path(image_arg: str) -> Path:
    candidates = [
        Path(image_arg),
        ROOT / image_arg,
        ROOT.parent / image_arg,
        resolve_data_root() / image_arg,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve image path: {image_arg}")


def infer_mask_dir(image_path: Path) -> Path | None:
    root = resolve_data_root().resolve()
    for dataset in Datasets.ALL_TRAIN.value:
        image_root = (root / dataset["images"]).resolve()
        if image_path.is_relative_to(image_root):
            if dataset["masks"] is None:
                return None
            return (root / dataset["masks"]).resolve()
    return None


def load_summary(summary_path: Path) -> dict[str, object]:
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_records(metrics_path: Path, predictions_dir: Path) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        image_path = Path(payload["image_path"]).resolve()
        prediction_path = predictions_dir / f"{safe_prediction_stem(image_path)}.png"
        if not prediction_path.exists():
            continue
        records.append(
            PredictionRecord(
                image_path=image_path,
                dataset_name=payload["dataset_name"],
                is_forged=bool(payload["is_forged"]),
                image_of1=float(payload["image_of1"]),
                pred_component_count=int(payload["pred_component_count"]),
                gt_component_count=int(payload["gt_component_count"]),
                pred_positive_rate=float(payload["pred_positive_rate"]),
                mean_refined_probability=float(payload["mean_refined_probability"]),
                prediction_path=prediction_path.resolve(),
            )
        )
    return records


def sort_records(records: list[PredictionRecord], sort_by: str) -> list[PredictionRecord]:
    if sort_by == "input":
        return records
    if sort_by == "worst-of1":
        return sorted(records, key=lambda record: (record.image_of1, -record.pred_component_count))
    if sort_by == "best-of1":
        return sorted(records, key=lambda record: (-record.image_of1, record.pred_component_count))
    if sort_by == "most-components":
        return sorted(records, key=lambda record: (-record.pred_component_count, record.image_of1))
    if sort_by == "highest-pred-pos":
        return sorted(records, key=lambda record: (-record.pred_positive_rate, record.image_of1))
    raise ValueError(f"Unsupported sort mode: {sort_by}")


def filter_records(records: list[PredictionRecord], args: argparse.Namespace) -> list[PredictionRecord]:
    if args.forged_only and args.authentic_only:
        raise ValueError("Choose at most one of --forged-only or --authentic-only.")

    filtered = records
    if args.forged_only:
        filtered = [record for record in filtered if record.is_forged]
    if args.authentic_only:
        filtered = [record for record in filtered if not record.is_forged]
    if args.image is not None:
        target_path = resolve_image_path(args.image)
        filtered = [record for record in filtered if record.image_path == target_path]
    return sort_records(filtered, args.sort_by)
def load_prediction_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return (mask > 0).astype(np.uint8)


def load_ground_truth_mask(record: PredictionRecord, size: tuple[int, int]) -> np.ndarray:
    if not record.is_forged:
        return np.zeros(size, dtype=np.uint8)

    mask_dir = infer_mask_dir(record.image_path)
    if mask_dir is None:
        return np.zeros(size, dtype=np.uint8)

    mask_path = mask_dir / record.image_path.name.replace(".png", ".npy")
    if not mask_path.exists():
        return np.zeros(size, dtype=np.uint8)

    mask = np.load(mask_path)
    mask = normalize_mask_array(mask)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
    mask_resized = np.asarray(
        mask_img.resize((size[1], size[0]), resample=Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    return (mask_resized > 0).astype(np.uint8)


def apply_overlay(image: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()

    pred_alpha = 0.42 * pred_mask[..., None].astype(np.float32)
    pred_color = np.array([1.0, 0.15, 0.15], dtype=np.float32).reshape(1, 1, 3)
    overlay = overlay * (1.0 - pred_alpha) + pred_color * pred_alpha

    gt_boundary = gt_mask.astype(np.uint8)
    if gt_boundary.any():
        shifts = [
            np.roll(gt_boundary, 1, axis=0),
            np.roll(gt_boundary, -1, axis=0),
            np.roll(gt_boundary, 1, axis=1),
            np.roll(gt_boundary, -1, axis=1),
        ]
        edge = gt_boundary.astype(bool)
        for shifted in shifts:
            edge &= shifted.astype(bool)
        edge = gt_boundary.astype(bool) & ~edge
        overlay[edge] = np.array([0.1, 1.0, 0.3], dtype=np.float32)

    return overlay


def render_records(records: list[PredictionRecord], image_size: tuple[int, int]):
    rows = len(records)
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, record in enumerate(records):
        image = load_display_image(record.image_path, image_size)
        pred_mask = load_prediction_mask(record.prediction_path)
        gt_mask = load_ground_truth_mask(record, pred_mask.shape)
        overlay = apply_overlay(image, pred_mask, gt_mask)

        panels = [
            (image, "Image", None),
            (pred_mask, f"Prediction ({record.pred_component_count} comps)", "gray"),
            (gt_mask, f"GT ({record.gt_component_count} comps)", "gray"),
            (overlay, f"Overlay oF1={record.image_of1:.3f}", None),
        ]

        for col_idx, (panel, title, cmap) in enumerate(panels):
            axis = axes[row_idx, col_idx]
            if cmap is None:
                axis.imshow(panel)
            else:
                axis.imshow(panel, cmap=cmap, vmin=0.0, vmax=1.0)
            axis.set_title(title)
            axis.axis("off")

        label = "forged" if record.is_forged else "authentic"
        axes[row_idx, 0].set_ylabel(
            f"{label}\n{record.image_path.name}\nref={record.mean_refined_probability:.3f}\npred_pos={record.pred_positive_rate:.2%}",
            rotation=90,
            fontsize=9,
        )

    fig.tight_layout()
    return fig


def list_records(records: list[PredictionRecord]) -> None:
    for idx, record in enumerate(records):
        label = "forged" if record.is_forged else "authentic"
        print(
            f"[{idx}] {label:<9} "
            f"oF1={record.image_of1:.3f} "
            f"pred_comp={record.pred_component_count:<3d} "
            f"pred_pos={record.pred_positive_rate:.2%} "
            f"{record.image_path}"
        )


def main() -> None:
    args = parse_args()

    eval_dir = args.eval_dir.resolve()
    metrics_path = resolve_optional_path(args.metrics_file, eval_dir / "per_sample_metrics.jsonl")
    predictions_dir = resolve_optional_path(args.predictions_dir, eval_dir / "predictions")
    summary_path = resolve_optional_path(args.summary_file, eval_dir / "summary.json")

    summary = load_summary(summary_path)
    image_size_value = summary.get("config", {}).get("image_size", 488)
    image_size = (int(image_size_value), int(image_size_value))

    records = load_records(metrics_path, predictions_dir)
    filtered = filter_records(records, args)

    if not filtered:
        raise RuntimeError("No saved predictions matched the requested filters.")

    if args.list:
        list_records(filtered)
        return

    if args.all:
        selected = filtered
    else:
        if args.index < 0 or args.index >= len(filtered):
            raise IndexError(f"--index {args.index} is out of range for {len(filtered)} filtered samples.")
        stop = min(args.index + args.count, len(filtered))
        selected = filtered[args.index:stop]

    fig = render_records(selected, image_size=image_size)

    if args.save is not None:
        save_path = resolve_optional_path(Path(args.save), ROOT / "tmp_selected_predictions.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"[selected-viewer] saved={save_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
