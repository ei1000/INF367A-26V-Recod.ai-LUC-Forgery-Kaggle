from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from dataset import Datasets
from pipeline import pipeline


def _parse_dataset(name: str) -> Datasets:
    normalized = name.strip().upper()
    try:
        return Datasets[normalized]
    except KeyError as exc:
        valid = ", ".join(member.name for member in Datasets)
        raise argparse.ArgumentTypeError(
            f"Unknown dataset '{name}'. Expected one of: {valid}"
        ) from exc


def _parse_pm_iters(values: list[int]) -> list[int]:
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one pm_iters value.")

    cleaned: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value <= 0:
            raise argparse.ArgumentTypeError("pm_iters values must be positive integers.")
        if value not in seen:
            cleaned.append(value)
            seen.add(value)
    return cleaned


def _default_run_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the same training recipe across multiple PatchMatch iteration counts. "
            "Each run is written to its own output directory."
        )
    )
    parser.add_argument(
        "--pm-iters",
        nargs="+",
        type=int,
        default=[32, 16, 8],
        help="PatchMatch iteration counts to compare. Default: 32 16 8",
    )
    parser.add_argument(
        "--dataset",
        default="ALL_TRAIN",
        help="Dataset enum name from dataset.Datasets. Default: ALL_TRAIN",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per ablation run.")
    parser.add_argument("--image-size", type=int, default=488, help="Square resize used by the pipeline.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size passed to the pipeline.")
    parser.add_argument(
        "--feature-backbone",
        choices=["cnn", "dino"],
        default="cnn",
        help="Feature backbone used during the ablation.",
    )
    parser.add_argument(
        "--cnn-backbone",
        choices=["simple", "pretrained"],
        default="simple",
        help="CNN backbone variant when feature-backbone=cnn.",
    )
    parser.add_argument(
        "--cnn-pretrained-model",
        default="vgg16_bn",
        help="Torchvision model name used when cnn-backbone=pretrained.",
    )
    parser.add_argument(
        "--cnn-feature-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable feature normalization for pretrained CNN features.",
    )
    parser.add_argument(
        "--use-dino-transform",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply DINO preprocessing inside the dataset transform path.",
    )
    parser.add_argument(
        "--dino-model-name",
        default="dinov2_vits14",
        help="Torch Hub model name used when feature-backbone=dino.",
    )
    parser.add_argument("--dino-proj-dim", type=int, default=64, help="Optional DINO projection dimension.")
    parser.add_argument(
        "--separate-transforms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep raw images for Zernike while normalizing backbone inputs separately.",
    )
    parser.add_argument("--pm-beta", type=float, default=1000.0, help="Soft-argmax temperature.")
    parser.add_argument("--pm-random-window", type=int, default=50, help="Random-search window size.")
    parser.add_argument(
        "--pm-use-non-local",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the non-local reset step between propagation rounds.",
    )
    parser.add_argument(
        "--pm-non-local-limit",
        type=float,
        default=25.0,
        help="Squared-distance threshold for the non-local reset step.",
    )
    parser.add_argument("--log-every", type=int, default=10, help="Batch logging frequency inside each run.")
    parser.add_argument(
        "--output-root",
        default="artifacts/pm_iters_ablation",
        help="Parent directory for all ablation outputs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional subdirectory name. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--save-predictions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save per-batch prediction tensors. Disabled by default to reduce I/O noise.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume from per-run checkpoints if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned runs without starting training.",
    )
    return parser


def _build_manifest(args: argparse.Namespace, dataset: Datasets, pm_iters: list[int]) -> dict:
    run_name = args.run_name or _default_run_name()
    root_dir = Path(args.output_root) / run_name
    runs = []
    for value in pm_iters:
        runs.append(
            {
                "name": f"pm_iters_{value}",
                "pm_iters": value,
                "output_dir": str(root_dir / f"pm_iters_{value}"),
                "status": "pending",
            }
        )

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "output_root": str(root_dir),
        "dataset": dataset.name,
        "config": {
            "epochs": args.epochs,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "feature_backbone": args.feature_backbone,
            "cnn_backbone": args.cnn_backbone,
            "cnn_pretrained_model": args.cnn_pretrained_model,
            "cnn_feature_norm": args.cnn_feature_norm,
            "use_dino_transform": args.use_dino_transform,
            "dino_model_name": args.dino_model_name,
            "dino_proj_dim": args.dino_proj_dim,
            "separate_transforms": args.separate_transforms,
            "pm_beta": args.pm_beta,
            "pm_random_window": args.pm_random_window,
            "pm_use_non_local": args.pm_use_non_local,
            "pm_non_local_limit": args.pm_non_local_limit,
            "log_every": args.log_every,
            "save_predictions": args.save_predictions,
            "resume": args.resume,
        },
        "runs": runs,
    }


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _print_plan(manifest: dict) -> None:
    print(f"[ablation] run_name={manifest['run_name']}")
    print(f"[ablation] output_root={manifest['output_root']}")
    print(f"[ablation] dataset={manifest['dataset']}")
    for run in manifest["runs"]:
        print(f"[ablation] pm_iters={run['pm_iters']} -> {run['output_dir']}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dataset = _parse_dataset(args.dataset)
    pm_iters = _parse_pm_iters(args.pm_iters)
    manifest = _build_manifest(args, dataset, pm_iters)

    _print_plan(manifest)

    if args.dry_run:
        return

    root_dir = Path(manifest["output_root"])
    root_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root_dir / "ablation_plan.json"
    _write_manifest(manifest_path, manifest)

    for run in manifest["runs"]:
        output_dir = Path(run["output_dir"])
        if output_dir.exists() and not args.resume:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --resume or choose a different --run-name."
            )

        print(f"[ablation] starting pm_iters={run['pm_iters']}")
        run["status"] = "running"
        _write_manifest(manifest_path, manifest)

        try:
            pipeline(
                datasets=dataset,
                image_size=args.image_size,
                epochs=args.epochs,
                feature_backbone=args.feature_backbone,
                use_dino_transform=args.use_dino_transform,
                batch_size=args.batch_size,
                dino_model_name=args.dino_model_name,
                dino_proj_dim=args.dino_proj_dim,
                cnn_backbone=args.cnn_backbone,
                cnn_pretrained_model=args.cnn_pretrained_model,
                cnn_feature_norm=args.cnn_feature_norm,
                separate_transforms=args.separate_transforms,
                pm_iters=run["pm_iters"],
                pm_beta=args.pm_beta,
                pm_random_window=args.pm_random_window,
                pm_use_non_local=args.pm_use_non_local,
                pm_non_local_limit=args.pm_non_local_limit,
                log_every=args.log_every,
                output_dir=output_dir,
                checkpoint_name="latest.pt",
                resume=args.resume,
                save_predictions=args.save_predictions,
            )
        except Exception:
            run["status"] = "failed"
            _write_manifest(manifest_path, manifest)
            raise

        run["status"] = "completed"
        _write_manifest(manifest_path, manifest)
        print(f"[ablation] finished pm_iters={run['pm_iters']}")


if __name__ == "__main__":
    main()
