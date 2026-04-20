from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from configs.baseline_config import set_seed
from dataset_utils import list_labeled_samples
from datasets.splits import make_grouped_stratified_splits
from engine.checkpointing import load_checkpoint
from engine.validation_inference import collect_validation_predictions, score_validation_predictions
from evaluate_validation_postprocess import (
    _build_sliding_window_fn,
    _build_val_loader,
    _resolve_device,
    config_from_checkpoint,
    validate_model_loading_allowed,
)
from models.dino_segmenter import DinoSegmenter
from util.pixelmapUtil import PixelMapUtil


LOCAL_HOLDOUT_WARNING = (
    "This evaluates the reserved local holdout test split. "
    "Use it for final local review, not tuning."
)


def validate_local_holdout_allowed(confirm_local_holdout: bool) -> None:
    if confirm_local_holdout:
        return
    raise RuntimeError(
        "Reserved local holdout evaluation is disabled unless --confirm-local-holdout is set. "
        "This script evaluates the reserved local holdout test split, so use it for final local "
        "review, not tuning."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot local holdout evaluation for the baseline model.")
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
    parser.add_argument(
        "--confirm-local-holdout",
        action="store_true",
        default=False,
        help="Confirm this intentional one-shot evaluation of the reserved local holdout split.",
    )
    parser.add_argument(
        "--allow-torch-hub",
        action="store_true",
        default=False,
        help="Allow model reconstruction through torch.hub and DinoSegmenter.from_official.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(LOCAL_HOLDOUT_WARNING)

    try:
        validate_local_holdout_allowed(args.confirm_local_holdout)
        validate_model_loading_allowed(args.allow_torch_hub)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    device = _resolve_device(args.device)
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
    holdout_samples = splits["test"]

    print(f"Checkpoint path: {args.checkpoint}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    if "kaggle_score" in checkpoint:
        print(f"Stored checkpoint validation score: {float(checkpoint['kaggle_score']):.4f}")
    else:
        print("Stored checkpoint validation score: unknown")
    print(f"Local holdout samples: {len(holdout_samples)}")

    holdout_loader = _build_val_loader(config, holdout_samples, device)
    pixel_util = PixelMapUtil()
    print(
        "Model reconstruction explicitly allowed: "
        "using torch.hub via DinoSegmenter.from_official(...) for local holdout evaluation."
    )
    model = DinoSegmenter.from_official(
        model_name=config.dino_model_name,
        embed_dim=config.dino_embed_dim,
        freeze_encoder=config.freeze_dino_encoder,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = collect_validation_predictions(
        model=model,
        val_loader=holdout_loader,
        val_samples=holdout_samples,
        device=device,
        inference_mode=config.validation_inference_mode,
        sliding_window_fn=_build_sliding_window_fn(config),
        probability_dtype=config.validation_probability_dtype,
        collect_masks=config.compute_pixel_f1,
        transfer_mode=config.validation_transfer_mode,
    )

    result = score_validation_predictions(
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

    print(f"Local holdout kaggle_score: {result['kaggle_score']:.4f}")
    print(
        "Sample counts: "
        f"total={result['num_samples']} "
        f"forged={result['num_forged']} "
        f"authentic={result['num_authentic']}"
    )
    if result.get("pixel_f1") is not None:
        print(f"Local holdout pixel_f1: {result['pixel_f1']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
