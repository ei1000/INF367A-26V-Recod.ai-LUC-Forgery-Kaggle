import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import ForgeryDataset, regular_transform
from feature_extractors.cnn_feature_extractor import PyramidFeatureExtractor, PretrainedBackboneExtractor
from feature_extractors.zernike_feature_extractor import PyramidZernikeExtractor, default_pq_list
from cross_scale_patternmatch.pixel_propagator import PixelPropagator


def parse_args():
    p = argparse.ArgumentParser(description="Verify offset behavior on authentic vs forged images.")
    default_data = Path(__file__).resolve().parent.parent / "data"
    p.add_argument("--data-root", type=Path, default=default_data)
    p.add_argument("--split", type=str, default="train", choices=["train", "supplement"])
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--feature-backbone", type=str, default="cnn", choices=["cnn", "dino"])
    p.add_argument("--cnn-backbone", type=str, default="pretrained", choices=["simple", "pretrained"])
    p.add_argument("--cnn-pretrained-model", type=str, default="vgg16_bn", choices=["vgg16_bn", "resnet18"])
    p.add_argument("--no-cnn-feature-norm", action="store_true")
    p.add_argument("--iters", type=int, default=24)
    p.add_argument("--random-window", type=int, default=50)
    p.add_argument("--betas", type=str, default="2.5,5,30")
    p.add_argument("--n-auth", type=int, default=20)
    p.add_argument("--n-forged", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dino-model-name", type=str, default="dinov2_vits14")
    p.add_argument("--dino-proj-dim", type=int, default=64)
    p.add_argument("--output", type=Path, default=Path("runs/verify_d2prl_alignment.json"))
    return p.parse_args()


def imagenet_normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def split_dirs(split: str):
    if split == "train":
        return "train_images", "train_masks"
    return "supplemental_images", "supplemental_masks"


def collect_images(folder: Path):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    files = []
    for ext in exts:
        files.extend(folder.glob(ext))
    return sorted(files)


def pick_subset(paths, n, rng):
    paths = list(paths)
    if len(paths) <= n:
        return sorted(paths)
    return sorted(rng.sample(paths, n))


def build_backbone(args, device: str):
    if args.feature_backbone == "dino":
        from feature_extractors.dino_feature_extractor import PyramidDinoFeatureExtractor

        model = PyramidDinoFeatureExtractor(
            model_name=args.dino_model_name,
            normalize_input=True,
            proj_dim=args.dino_proj_dim,
        ).to(device)
        model.eval()
        return model

    if args.cnn_backbone == "pretrained":
        backbone = PretrainedBackboneExtractor(
            model_name=args.cnn_pretrained_model,
            out_dim=32,
            freeze=True,
        )
        model = PyramidFeatureExtractor(backbone=backbone).to(device)
        model.eval()
        return model

    model = PyramidFeatureExtractor().to(device)
    model.eval()
    return model


def build_dataset(args):
    image_dir, mask_dir = split_dirs(args.split)
    image_root = args.data_root / image_dir
    mask_root = args.data_root / mask_dir
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory not found: {image_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_root}")
    rng = random.Random(args.seed)

    auth_paths = collect_images(image_root / "authentic")
    forged_paths = collect_images(image_root / "forged")
    if len(auth_paths) == 0 or len(forged_paths) == 0:
        raise RuntimeError(
            f"No images found under {image_root / 'authentic'} or {image_root / 'forged'}."
        )
    sel_auth = pick_subset(auth_paths, args.n_auth, rng)
    sel_forged = pick_subset(forged_paths, args.n_forged, rng)

    metas = []
    for p in sel_auth:
        metas.append({"group": "authentic", "path": p})
    for p in sel_forged:
        metas.append({"group": "forged", "path": p})

    samples = [(m["path"], 0 if m["group"] == "authentic" else 1) for m in metas]
    ds = ForgeryDataset(
        samples=samples,
        mask_dir=mask_root,
        size=args.image_size,
        transform=regular_transform,
    )
    return ds, metas


def offset_stats(offsets: torch.Tensor, mask: torch.Tensor):
    dx = offsets[0].detach().float().cpu()
    dy = offsets[1].detach().float().cpu()
    mag = torch.sqrt(dx * dx + dy * dy)
    near_zero_ratio = float(((dx.abs() + dy.abs()) < 1.0).float().mean().item())
    mean_mag = float(mag.mean().item())
    p90_mag = float(torch.quantile(mag.flatten(), 0.9).item())

    gap = None
    if mask is not None:
        m = (mask.detach().cpu() > 0)
        if m.any() and (~m).any():
            gap = float(mag[m].mean().item() - mag[~m].mean().item())

    return {
        "near_zero_ratio": near_zero_ratio,
        "mean_mag": mean_mag,
        "p90_mag": p90_mag,
        "mask_bg_gap": gap,
    }


def aggregate_metric_dicts(metric_list):
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in metric_list if m[k] is not None]
        out[k] = float(sum(vals) / len(vals)) if vals else None
    return out


def run():
    args = parse_args()
    betas = [float(x.strip()) for x in args.betas.split(",") if x.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset, metas = build_dataset(args)
    backbone = build_backbone(args, device)
    zernike = PyramidZernikeExtractor(default_pq_list(max_order=5), kernel_size=13).to(device)
    zernike.eval()

    results = {
        "config": {
            "split": args.split,
            "image_size": args.image_size,
            "feature_backbone": args.feature_backbone,
            "cnn_backbone": args.cnn_backbone,
            "cnn_pretrained_model": args.cnn_pretrained_model,
            "cnn_feature_norm": (not args.no_cnn_feature_norm),
            "iters": args.iters,
            "random_window": args.random_window,
            "n_auth": args.n_auth,
            "n_forged": args.n_forged,
            "device": device,
            "betas": betas,
        },
        "summary": {},
    }

    for beta in betas:
        per_group = {
            "cnn": {"authentic": [], "forged": []},
            "zernike": {"authentic": [], "forged": []},
        }

        print(f"[verify] beta={beta}")
        for i in range(len(dataset)):
            img, mask, _ = dataset[i]
            group = metas[i]["group"]
            images = img.unsqueeze(0).to(device)
            mask = mask.to(device)

            images_backbone = images
            if args.feature_backbone == "cnn" and args.cnn_backbone == "pretrained":
                images_backbone = imagenet_normalize_tensor(images)

            with torch.no_grad():
                cnn_feats = backbone(images_backbone)
                if args.feature_backbone == "cnn" and args.cnn_backbone == "pretrained" and (not args.no_cnn_feature_norm):
                    cnn_feats = tuple(F.normalize(f, p=2, dim=1) for f in cnn_feats)
                zm_feats = zernike(images)

            propagator = PixelPropagator(
                images[0],
                tuple(f[0] for f in cnn_feats),
                tuple(f[0] for f in zm_feats),
                random_window=args.random_window,
            )
            cnn_offsets, z_offsets = propagator.propagation_layer(iters=args.iters, beta=beta)

            per_group["cnn"][group].append(offset_stats(cnn_offsets, mask))
            per_group["zernike"][group].append(offset_stats(z_offsets, mask))

            del cnn_feats, zm_feats, cnn_offsets, z_offsets, propagator, images, images_backbone
            if device == "cuda":
                torch.cuda.empty_cache()

        results["summary"][str(beta)] = {
            "cnn": {
                "authentic": aggregate_metric_dicts(per_group["cnn"]["authentic"]),
                "forged": aggregate_metric_dicts(per_group["cnn"]["forged"]),
            },
            "zernike": {
                "authentic": aggregate_metric_dicts(per_group["zernike"]["authentic"]),
                "forged": aggregate_metric_dicts(per_group["zernike"]["forged"]),
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"[verify] wrote summary to {args.output}")


if __name__ == "__main__":
    run()
