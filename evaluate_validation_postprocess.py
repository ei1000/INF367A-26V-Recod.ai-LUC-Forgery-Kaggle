from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from functools import partial
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import torch
from torch.utils.data import DataLoader

from configs.baseline_config import BaselineConfig, seed_worker, set_seed
from dataset_utils import list_labeled_samples
from datasets.forgery_dataset import ForgeryDataset
from datasets.splits import make_grouped_stratified_splits
from engine.checkpointing import load_checkpoint
from engine.validation_inference import collect_validation_predictions, score_validation_predictions
from inference.sliding_window_dino import sliding_window_dino
from models.dino_segmenter import DinoSegmenter
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


def config_from_checkpoint(checkpoint: dict) -> BaselineConfig:
    raw_config = _as_mapping(checkpoint.get("config"))
    valid_field_names = {field.name for field in fields(BaselineConfig)}
    filtered = {key: value for key, value in raw_config.items() if key in valid_field_names}
    return BaselineConfig(**filtered)


def _parse_list(text: str, parser) -> list:
    items = []
    for raw_item in text.split(","):
        item = raw_item.strip()
        if item == "":
            raise ValueError("Empty list items are not allowed")
        items.append(parser(item))
    return items


def parse_float_list(text: str) -> list[float]:
    return _parse_list(text, float)


def parse_int_list(text: str) -> list[int]:
    return _parse_list(text, int)


def parse_bool_list(text: str) -> list[bool]:
    truthy = {"1", "true", "t", "yes", "y", "on"}
    falsy = {"0", "false", "f", "no", "n", "off"}

    def parse_bool(item: str) -> bool:
        lowered = item.lower()
        if lowered in truthy:
            return True
        if lowered in falsy:
            return False
        raise ValueError(f"Expected a boolean value, got {item!r}")

    return _parse_list(text, parse_bool)


def parse_optional_float_list(text: str) -> list[float | None]:
    def parse_optional_float(item: str) -> float | None:
        if item.lower() == "none":
            return None
        return float(item)

    return _parse_list(text, parse_optional_float)


def _namespace_get(args, name: str):
    if isinstance(args, dict):
        return args[name]
    return getattr(args, name)


def _unique_preserve_order(values: Iterable) -> list:
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def iter_postprocess_settings(config: BaselineConfig, args) -> list[dict]:
    pred_thresholds = _namespace_get(args, "pred_thresholds")
    min_component_areas = _namespace_get(args, "min_component_areas")
    confident_thresholds = _namespace_get(args, "confident_thresholds")
    smooth_options = _namespace_get(args, "smooth_options")
    opening_options = _namespace_get(args, "opening_options")
    closing_options = _namespace_get(args, "closing_options")
    fill_holes_options = _namespace_get(args, "fill_holes_options")
    keep_confident_seeded_options = _namespace_get(args, "keep_confident_seeded_options")
    max_settings = _namespace_get(args, "max_settings")

    if max_settings is not None and max_settings <= 0:
        return []

    settings: list[dict] = []
    for (
        pred_threshold,
        min_component_area,
        confident_threshold,
        smooth_probabilities,
        apply_opening,
        apply_closing,
        fill_holes,
        keep_confident_seeded_components,
    ) in product(
        _unique_preserve_order(pred_thresholds),
        _unique_preserve_order(min_component_areas),
        _unique_preserve_order(confident_thresholds),
        _unique_preserve_order(smooth_options),
        _unique_preserve_order(opening_options),
        _unique_preserve_order(closing_options),
        _unique_preserve_order(fill_holes_options),
        _unique_preserve_order(keep_confident_seeded_options),
    ):
        settings.append(
            {
                "pred_threshold": pred_threshold,
                "min_component_area": min_component_area,
                "confident_threshold": confident_threshold,
                "smooth_probabilities": smooth_probabilities,
                "apply_opening": apply_opening,
                "apply_closing": apply_closing,
                "fill_holes": fill_holes,
                "keep_confident_seeded_components": keep_confident_seeded_components,
                "harden_temperature": config.harden_temperature,
                "hard_clip_low": config.hard_clip_low,
                "hard_clip_high": config.hard_clip_high,
                "compute_pixel_f1": config.compute_pixel_f1,
                "verify_score_equivalence": False,
            }
        )
        if max_settings is not None and len(settings) >= max_settings:
            break
    return settings


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def validate_model_loading_allowed(allow_torch_hub: bool) -> None:
    if allow_torch_hub:
        return
    raise RuntimeError(
        "Model reconstruction is disabled unless --allow-torch-hub is set. "
        "This script rebuilds DinoSegmenter via torch.hub through "
        "DinoSegmenter.from_official(...), which may initialize or download DINO code/weights "
        "if they are not already cached. Enable it only for an intentional trusted local run."
    )


def validate_reproduced_validation_score(
    stored_score: float | None,
    reproduced_score: float,
    tolerance: float,
    allow_mismatch: bool,
) -> None:
    if tolerance < 0:
        raise ValueError(f"score tolerance must be non-negative, got {tolerance}")
    if stored_score is None:
        return

    score_delta = abs(float(stored_score) - float(reproduced_score))
    if score_delta <= tolerance:
        return

    message = (
        "Reproduced validation score does not match the checkpoint's stored validation score: "
        f"stored={float(stored_score):.6f}, reproduced={float(reproduced_score):.6f}, "
        f"delta={score_delta:.6f}, tolerance={tolerance:.6f}. "
        "Check split/config reconstruction before trusting post-processing sweep results."
    )
    if allow_mismatch:
        print(f"WARNING: {message}")
        return
    raise RuntimeError(f"{message} Use --allow-score-mismatch to continue anyway.")


def _build_val_loader(config: BaselineConfig, val_samples, device: torch.device) -> DataLoader:
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


def _build_sliding_window_fn(config: BaselineConfig):
    sw_patch = config.sliding_window_size or config.target_size
    sw_stride = config.sliding_stride or (sw_patch // 2)
    return partial(
        sliding_window_dino,
        patch_size=sw_patch,
        stride=sw_stride,
        batch_size=config.sliding_batch_size,
    )


def _format_optional_float(value: float | None) -> str:
    return "none" if value is None else f"{value:g}"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _print_ranked_rows(rows: list[dict]) -> None:
    headers = [
        "rank",
        "kaggle_score",
        "pred_threshold",
        "min_component_area",
        "smooth",
        "closing",
        "opening",
        "fill_holes",
        "keep_confident_seeded",
        "confident_threshold",
    ]
    widths = {
        "rank": 4,
        "kaggle_score": 12,
        "pred_threshold": 14,
        "min_component_area": 18,
        "smooth": 6,
        "closing": 7,
        "opening": 7,
        "fill_holes": 10,
        "keep_confident_seeded": 21,
        "confident_threshold": 19,
    }
    print(" ".join(f"{header:<{widths[header]}}" for header in headers))
    for idx, row in enumerate(sorted(rows, key=lambda item: (-item["kaggle_score"], item["rank"])), start=1):
        print(
            f"{idx:<{widths['rank']}} "
            f"{row['kaggle_score']:<{widths['kaggle_score']}.4f} "
            f"{row['pred_threshold']:<{widths['pred_threshold']}.4f} "
            f"{row['min_component_area']:<{widths['min_component_area']}} "
            f"{_bool_text(row['smooth_probabilities']):<{widths['smooth']}} "
            f"{_bool_text(row['apply_closing']):<{widths['closing']}} "
            f"{_bool_text(row['apply_opening']):<{widths['opening']}} "
            f"{_bool_text(row['fill_holes']):<{widths['fill_holes']}} "
            f"{_bool_text(row['keep_confident_seeded_components']):<{widths['keep_confident_seeded']}} "
            f"{_format_optional_float(row['confident_threshold']):<{widths['confident_threshold']}}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate validation post-processing on cached probabilities.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/checkpoints/best_by_kaggle_score.pt"),
        help="Trusted local checkpoint to evaluate.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use: auto, cpu, cuda, or a specific torch device string.",
    )
    parser.add_argument("--pred-thresholds", default=None, help="Comma-separated prediction thresholds.")
    parser.add_argument(
        "--min-component-areas",
        default=None,
        help="Comma-separated min component areas.",
    )
    parser.add_argument(
        "--confident-thresholds",
        default=None,
        help="Comma-separated confident thresholds or 'none'.",
    )
    parser.add_argument("--smooth-options", default=None, help="Comma-separated booleans.")
    parser.add_argument("--opening-options", default=None, help="Comma-separated booleans.")
    parser.add_argument("--closing-options", default=None, help="Comma-separated booleans.")
    parser.add_argument("--fill-holes-options", default=None, help="Comma-separated booleans.")
    parser.add_argument(
        "--keep-confident-seeded-options",
        default=None,
        help="Comma-separated booleans.",
    )
    parser.add_argument(
        "--allow-torch-hub",
        action="store_true",
        default=False,
        help="Allow model reconstruction through torch.hub and DinoSegmenter.from_official.",
    )
    parser.add_argument(
        "--score-tolerance",
        type=float,
        default=1e-4,
        help="Allowed absolute difference between stored and reproduced validation kaggle_score.",
    )
    parser.add_argument(
        "--allow-score-mismatch",
        action="store_true",
        default=False,
        help="Warn instead of stopping when reproduced validation score differs from the checkpoint.",
    )
    parser.add_argument("--max-settings", type=int, default=None, help="Optional cap on sweep combinations.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device = _resolve_device(args.device)

    print("Using validation split for tuning; local test split is not evaluated by this script.")
    print(f"Using device: {device}")

    # Trusted local checkpoints only: torch.load uses pickle under the hood.
    checkpoint = load_checkpoint(args.checkpoint, map_location=device, trusted=True)
    config = config_from_checkpoint(checkpoint)

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
        print(f"Debug val_subset enabled: using {config.val_subset} of {len(val_samples)} val samples")
        val_samples = val_samples[: config.val_subset]

    print(f"Validation samples: {len(val_samples)}")
    print(f"Checkpoint path: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    if "kaggle_score" in checkpoint:
        print(f"Stored checkpoint validation score: {float(checkpoint['kaggle_score']):.4f}")

    val_loader = _build_val_loader(config, val_samples, device)
    pixel_util = PixelMapUtil()
    try:
        validate_model_loading_allowed(args.allow_torch_hub)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "Model reconstruction explicitly allowed: "
        "using torch.hub via DinoSegmenter.from_official(...) for local evaluation."
    )
    model = DinoSegmenter.from_official(
        model_name=config.dino_model_name,
        embed_dim=config.dino_embed_dim,
        freeze_encoder=config.freeze_dino_encoder,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sliding_window_fn = _build_sliding_window_fn(config)
    predictions = collect_validation_predictions(
        model=model,
        val_loader=val_loader,
        val_samples=val_samples,
        device=device,
        inference_mode=config.validation_inference_mode,
        sliding_window_fn=sliding_window_fn,
        probability_dtype=config.validation_probability_dtype,
        collect_masks=config.compute_pixel_f1,
        transfer_mode=config.validation_transfer_mode,
    )

    baseline_result = score_validation_predictions(
        predictions=predictions,
        pixel_util=pixel_util,
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
    print(f"Reproduced validation score: {baseline_result['kaggle_score']:.4f}")
    stored_validation_score = checkpoint.get("kaggle_score")
    try:
        validate_reproduced_validation_score(
            stored_score=stored_validation_score,
            reproduced_score=baseline_result["kaggle_score"],
            tolerance=args.score_tolerance,
            allow_mismatch=args.allow_score_mismatch,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if stored_validation_score is not None:
        print(
            "Stored/reproduced validation delta: "
            f"{abs(float(stored_validation_score) - baseline_result['kaggle_score']):.6f}"
        )

    if args.pred_thresholds is None:
        pred_thresholds = [config.pred_threshold]
    else:
        pred_thresholds = parse_float_list(args.pred_thresholds)
    if args.min_component_areas is None:
        min_component_areas = [config.min_component_area]
    else:
        min_component_areas = parse_int_list(args.min_component_areas)
    if args.confident_thresholds is None:
        confident_thresholds = [config.post_process_confident_threshold]
    else:
        confident_thresholds = parse_optional_float_list(args.confident_thresholds)
    if args.smooth_options is None:
        smooth_options = [config.post_process_smooth_probabilities]
    else:
        smooth_options = parse_bool_list(args.smooth_options)
    if args.opening_options is None:
        opening_options = [config.post_process_apply_opening]
    else:
        opening_options = parse_bool_list(args.opening_options)
    if args.closing_options is None:
        closing_options = [config.post_process_apply_closing]
    else:
        closing_options = parse_bool_list(args.closing_options)
    if args.fill_holes_options is None:
        fill_holes_options = [config.post_process_fill_holes]
    else:
        fill_holes_options = parse_bool_list(args.fill_holes_options)
    if args.keep_confident_seeded_options is None:
        keep_confident_seeded_options = [config.post_process_keep_confident_seeded_components]
    else:
        keep_confident_seeded_options = parse_bool_list(args.keep_confident_seeded_options)

    sweep_args = SimpleNamespace(
        pred_thresholds=pred_thresholds,
        min_component_areas=min_component_areas,
        confident_thresholds=confident_thresholds,
        smooth_options=smooth_options,
        opening_options=opening_options,
        closing_options=closing_options,
        fill_holes_options=fill_holes_options,
        keep_confident_seeded_options=keep_confident_seeded_options,
        max_settings=args.max_settings,
    )
    sweep_settings = iter_postprocess_settings(config, sweep_args)

    sweep_rows: list[dict] = []
    for rank, setting in enumerate(sweep_settings, start=1):
        result = score_validation_predictions(
            predictions=predictions,
            pixel_util=pixel_util,
            pred_threshold=setting["pred_threshold"],
            harden_temperature=setting["harden_temperature"],
            hard_clip_low=setting["hard_clip_low"],
            hard_clip_high=setting["hard_clip_high"],
            min_component_area=setting["min_component_area"],
            compute_pixel_f1=setting["compute_pixel_f1"],
            verify_score_equivalence=setting["verify_score_equivalence"],
            confident_threshold=setting["confident_threshold"],
            smooth_probabilities=setting["smooth_probabilities"],
            fill_holes=setting["fill_holes"],
            apply_opening=setting["apply_opening"],
            apply_closing=setting["apply_closing"],
            keep_confident_seeded_components=setting["keep_confident_seeded_components"],
        )
        sweep_rows.append({"rank": rank, **setting, **result})

    print("Sweep results:")
    if sweep_rows:
        _print_ranked_rows(sweep_rows)
    else:
        print("No sweep settings generated.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
