from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "View saved prediction tensors from artifacts/predictions. "
            "These files contain mask batches only; they do not store source image paths."
        )
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="artifacts/predictions",
        help="Prediction .pt file or a directory containing prediction files.",
    )
    parser.add_argument(
        "--file-index",
        type=int,
        default=0,
        help="If source is a directory, choose which .pt file to open after sorting by name.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="If source is a directory, open the last .pt file instead of using --file-index.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List prediction files in the directory and exit.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Mask index inside the loaded batch tensor.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render every mask in the loaded batch as a grid.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional image path to overlay the selected mask on top of.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help="Overlay opacity when --image is provided.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the figure as an image instead of only showing it.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip plt.show(); useful together with --save.",
    )
    return parser


def _resolve_prediction_file(source: Path, file_index: int, latest: bool, list_only: bool) -> Path | None:
    if source.is_file():
        return source

    if not source.exists():
        raise FileNotFoundError(f"Could not find source path: {source}")

    files = sorted(source.rglob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt prediction files found under: {source}")

    if list_only:
        for idx, path in enumerate(files):
            print(f"[{idx}] {path}")
        return None

    if latest:
        return files[-1]

    if file_index < 0 or file_index >= len(files):
        raise IndexError(f"--file-index {file_index} is out of range for {len(files)} files.")
    return files[file_index]


def _load_masks(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {path}, got {type(tensor)}")

    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 4 and tensor.shape[1] == 1:
        tensor = tensor.squeeze(1)
    elif tensor.dim() != 3:
        raise ValueError(f"Expected mask tensor of shape [H,W], [B,H,W], or [B,1,H,W], got {tuple(tensor.shape)}")

    return tensor


def _load_overlay_image(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = plt.imread(path)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]

    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    image_tensor = F.interpolate(image_tensor, size=size, mode="bilinear", align_corners=False)
    image = image_tensor.squeeze(0).permute(1, 2, 0).numpy()

    if image.max() > 1.0:
        image = image / 255.0
    return image


def _mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
    return mask.detach().cpu().numpy()


def _render_single_mask(mask: torch.Tensor, title: str, image: np.ndarray | None, alpha: float):
    fig, ax = plt.subplots(figsize=(6, 6))
    mask_np = _mask_to_numpy(mask)

    if image is not None:
        ax.imshow(image)
        overlay = np.ma.masked_where(mask_np <= 0, mask_np)
        ax.imshow(overlay, cmap="Reds", alpha=alpha, interpolation="nearest")
    else:
        ax.imshow(mask_np, cmap="gray", interpolation="nearest")

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def _render_grid(masks: torch.Tensor, title_prefix: str):
    count = masks.shape[0]
    cols = min(4, count)
    rows = math.ceil(count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(-1)

    for idx, ax in enumerate(axes):
        if idx < count:
            ax.imshow(_mask_to_numpy(masks[idx]), cmap="gray", interpolation="nearest")
            ax.set_title(f"{title_prefix} [{idx}]")
        ax.axis("off")

    fig.tight_layout()
    return fig


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    source = (ROOT / args.source).resolve() if not Path(args.source).is_absolute() else Path(args.source)
    prediction_file = _resolve_prediction_file(source, args.file_index, args.latest, args.list)
    if prediction_file is None:
        return

    masks = _load_masks(prediction_file)
    print(f"[viewer] file={prediction_file}")
    print(f"[viewer] shape={tuple(masks.shape)} dtype={masks.dtype}")

    if args.all:
        fig = _render_grid(masks, prediction_file.name)
    else:
        if args.index < 0 or args.index >= masks.shape[0]:
            raise IndexError(f"--index {args.index} is out of range for batch size {masks.shape[0]}")
        image = None
        if args.image is not None:
            image_path = (ROOT / args.image).resolve() if not Path(args.image).is_absolute() else Path(args.image)
            image = _load_overlay_image(image_path, tuple(masks.shape[-2:]))
        fig = _render_single_mask(
            masks[args.index],
            title=f"{prediction_file.name} [{args.index}]",
            image=image,
            alpha=args.alpha,
        )

    if args.save is not None:
        save_path = (ROOT / args.save).resolve() if not Path(args.save).is_absolute() else Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[viewer] saved={save_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
