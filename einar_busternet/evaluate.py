from __future__ import annotations

import argparse
import json
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from dataset_utils import list_labeled_samples
from datasets.forgery_dataset import ForgeryDataset
from datasets.splits import make_grouped_stratified_splits
from einar_busternet.config import BusterNetConfig, seed_worker, set_seed
from einar_busternet.model import BusterNetUnionWrapper
from engine.checkpointing import load_checkpoint
from engine.validation_inference import collect_validation_predictions, score_validation_predictions
from einar_busternet.train import _build_sliding_window_fn, build_model
from util.pixelmapUtil import PixelMapUtil


def _as_mapping(config_obj) -> dict:
    if config_obj is None:
        return {}
    if isinstance(config_obj, dict):
        return dict(config_obj)
    if is_dataclass(config_obj):
        return {field.name: getattr(config_obj, field.name) for field in fields(config_obj)}
    if hasattr(config_obj, "__dict__"):
        return dict(vars(config_obj))
    return {}


def config_from_checkpoint(checkpoint: dict) -> BusterNetConfig:
    raw_config = _as_mapping(checkpoint.get("config"))
    valid_field_names = {field.name for field in fields(BusterNetConfig)}
    filtered = {key: value for key, value in raw_config.items() if key in valid_field_names}
    return BusterNetConfig(**filtered)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DINO-BusterNet checkpoint on validation.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("einar_busternet/artifacts/checkpoints/best.pt"),
        help="Trusted local BusterNet checkpoint to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("einar_busternet/artifacts/results/eval_summary.json"),
        help="Where to write the evaluation summary JSON.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or a specific torch device string.",
    )
    parser.add_argument("--val-subset", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--pred-threshold", type=float, default=None)
    parser.add_argument(
        "--validation-transfer-mode",
        choices=("per_batch", "accumulate_gpu"),
        default=None,
    )
    parser.add_argument(
        "--allow-torch-hub",
        action="store_true",
        default=False,
        help="Allow model reconstruction through torch.hub and BusterNet model builders.",
    )
    return parser.parse_args(argv)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def validate_model_loading_allowed(allow_torch_hub: bool) -> None:
    if allow_torch_hub:
        return
    # Avoid surprising network/cache side effects unless evaluation explicitly allows it.
    raise RuntimeError(
        "Model reconstruction is disabled unless --allow-torch-hub is set. "
        "This rebuilds BusterNet through torch.hub, which may initialize or download "
        "DINO code/weights if they are not cached."
    )


def _build_val_loader(config: BusterNetConfig, val_samples, device: torch.device) -> DataLoader:
    val_loader_generator = torch.Generator()
    val_loader_generator.manual_seed(config.seed + 1)
    return DataLoader(
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


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_model_loading_allowed(args.allow_torch_hub)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    device = _resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device, trusted=True)
    config = config_from_checkpoint(checkpoint)
    overrides = {}
    if args.val_subset is not None:
        overrides["val_subset"] = args.val_subset
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.pred_threshold is not None:
        overrides["pred_threshold"] = args.pred_threshold
    if args.validation_transfer_mode is not None:
        overrides["validation_transfer_mode"] = args.validation_transfer_mode
    if overrides:
        config = replace(config, **overrides)

    set_seed(config.seed)
    all_samples = list_labeled_samples(Path(config.data_root))
    splits = make_grouped_stratified_splits(
        all_samples,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )
    val_samples = splits["val"]
    if config.val_subset is not None:
        val_samples = val_samples[: config.val_subset]

    print(f"Using device: {device}")
    print(f"Checkpoint path: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    if "kaggle_score" in checkpoint:
        print(f"Stored checkpoint validation score: {float(checkpoint['kaggle_score']):.4f}")
    print(f"Validation samples: {len(val_samples)}")

    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    wrapped_model = BusterNetUnionWrapper(model, eps=config.union_wrapper_eps)

    val_loader = _build_val_loader(config, val_samples, device)
    predictions = collect_validation_predictions(
        model=wrapped_model,
        val_loader=val_loader,
        val_samples=val_samples,
        device=device,
        inference_mode=config.validation_inference_mode,
        sliding_window_fn=_build_sliding_window_fn(config),
        probability_dtype=config.validation_probability_dtype,
        collect_masks=config.compute_pixel_f1,
        transfer_mode=config.validation_transfer_mode,
    )

    result = score_validation_predictions(
        predictions=predictions,
        pixel_util=PixelMapUtil(),
        pred_threshold=config.pred_threshold,
        harden_temperature=config.harden_temperature,
        hard_clip_low=config.hard_clip_low,
        hard_clip_high=config.hard_clip_high,
        min_component_area=config.min_component_area,
        compute_pixel_f1=config.compute_pixel_f1,
        verify_score_equivalence=config.verify_score_equivalence,
        confident_threshold=config.post_process_confident_threshold,
        smooth_probabilities=config.post_process_smooth_probabilities,
        fill_holes=config.post_process_fill_holes,
        apply_opening=config.post_process_apply_opening,
        apply_closing=config.post_process_apply_closing,
        keep_confident_seeded_components=config.post_process_keep_confident_seeded_components,
    )
    result.update(
        {
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "stored_checkpoint_kaggle_score": checkpoint.get("kaggle_score"),
            "validation_transfer_mode": config.validation_transfer_mode,
            "validation_inference_mode": config.validation_inference_mode,
            "probability_dtype": config.validation_probability_dtype,
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")

    print(f"Validation kaggle_score: {result['kaggle_score']:.4f}")
    print(
        "Sample counts: "
        f"total={result['num_samples']} "
        f"forged={result['num_forged']} "
        f"authentic={result['num_authentic']}"
    )
    if result.get("pixel_f1") is not None:
        print(f"Validation pixel_f1: {result['pixel_f1']:.4f}")
    print(f"Wrote summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
