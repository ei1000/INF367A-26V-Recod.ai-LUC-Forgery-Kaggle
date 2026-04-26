from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


@dataclass
class Component:
    channel_idx: int
    component_idx: int
    mask: np.ndarray
    area: int
    bbox: tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize connected mask components to inspect whether they look like "
            "source/target pairs or separate objects."
        )
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=DEFAULT_DATA_DIR / "train_masks",
        help="Directory containing .npy mask files.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_DATA_DIR / "train_images",
        help="Image directory. Supports train_images with forged/authentic subfolders.",
    )
    parser.add_argument(
        "--case-ids",
        nargs="*",
        default=None,
        help="Specific case IDs to inspect, without extension.",
    )
    parser.add_argument(
        "--min-components",
        type=int,
        default=2,
        help="When auto-selecting, require at least this many connected components.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="How many cases to render when auto-selecting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "mask_component_inspection",
        help="Where to save rendered figures.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=24,
        help="Extra pixels around each component crop.",
    )
    return parser.parse_args()


def resolve_image_path(case_id: str, image_dir: Path) -> Path:
    candidates = [
        image_dir / "forged" / f"{case_id}.png",
        image_dir / "authentic" / f"{case_id}.png",
        image_dir / f"{case_id}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find image for case_id={case_id}: {candidates}")


def load_grayscale_image(image_path: Path) -> np.ndarray:
    image = np.array(Image.open(image_path))
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)
    return image.astype(np.float32)


def infer_channel_masks(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask)
    if mask.ndim == 2:
        return [(mask > 0).astype(np.uint8)]
    if mask.ndim != 3:
        raise ValueError(f"Unsupported mask shape {mask.shape}")

    if mask.shape[0] <= 16 and mask.shape[0] <= mask.shape[-1] and mask.shape[0] <= mask.shape[-2]:
        return [(mask[i] > 0).astype(np.uint8) for i in range(mask.shape[0])]
    if mask.shape[-1] <= 16 and mask.shape[-1] <= mask.shape[0] and mask.shape[-1] <= mask.shape[1]:
        return [(mask[..., i] > 0).astype(np.uint8) for i in range(mask.shape[-1])]
    raise ValueError(f"Could not infer channel axis for mask with shape {mask.shape}")


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def extract_components(mask: np.ndarray) -> tuple[list[np.ndarray], list[Component]]:
    channel_masks = infer_channel_masks(mask)
    components: list[Component] = []
    for channel_idx, channel_mask in enumerate(channel_masks):
        labeled, count = ndi.label(channel_mask > 0)
        for component_idx in range(1, count + 1):
            component = (labeled == component_idx).astype(np.uint8)
            if not component.any():
                continue
            components.append(
                Component(
                    channel_idx=channel_idx,
                    component_idx=component_idx,
                    mask=component,
                    area=int(component.sum()),
                    bbox=bbox_from_mask(component),
                )
            )
    return channel_masks, components


def make_component_map(components: list[Component], shape: tuple[int, int]) -> np.ndarray:
    component_map = np.zeros(shape, dtype=np.int32)
    for idx, component in enumerate(components, start=1):
        component_map[component.mask.astype(bool)] = idx
    return component_map


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    low = float(image.min())
    high = float(image.max())
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return (image - low) / (high - low)


def apply_overlay(image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    base = np.repeat(normalize_image(image)[..., None], 3, axis=2)
    alpha = 0.45 * (mask > 0)[..., None].astype(np.float32)
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    return base * (1.0 - alpha) + color_arr * alpha


def expand_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int], margin: int) -> tuple[int, int, int, int]:
    y0, x0, y1, x1 = bbox
    height, width = shape
    return (
        max(0, y0 - margin),
        max(0, x0 - margin),
        min(height, y1 + margin),
        min(width, x1 + margin),
    )


def crop_region(image: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    y0, x0, y1, x1 = bbox
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def build_similarity_table(image: np.ndarray, components: list[Component], margin: int) -> list[tuple[float, int, int]]:
    if len(components) < 2:
        return []

    similarities: list[tuple[float, int, int]] = []
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            bbox_i = expand_bbox(components[i].bbox, image.shape, margin)
            bbox_j = expand_bbox(components[j].bbox, image.shape, margin)
            crop_i, mask_i = crop_region(image, components[i].mask, bbox_i)
            crop_j, mask_j = crop_region(image, components[j].mask, bbox_j)
            patch_i = masked_patch_signature(crop_i, mask_i)
            patch_j = masked_patch_signature(crop_j, mask_j)
            similarity = cosine_similarity(patch_i, patch_j)
            similarities.append((similarity, i, j))
    similarities.sort(reverse=True)
    return similarities[:5]


def masked_patch_signature(crop: np.ndarray, mask: np.ndarray, size: int = 64) -> np.ndarray:
    crop = crop.astype(np.float32)
    mask = (mask > 0).astype(np.float32)
    if not mask.any():
        return np.zeros(size * size, dtype=np.float32)

    crop = normalize_image(crop)
    masked = crop * mask
    pil = Image.fromarray((masked * 255).astype(np.uint8))
    resized = np.array(pil.resize((size, size), resample=Image.BILINEAR), dtype=np.float32) / 255.0
    vec = resized.reshape(-1)
    vec = vec - vec.mean()
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return np.zeros_like(vec)
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b))


def render_case(case_id: str, image_dir: Path, mask_dir: Path, output_dir: Path, margin: int) -> Path:
    image_path = resolve_image_path(case_id, image_dir=image_dir)
    mask_path = mask_dir / f"{case_id}.npy"
    if not mask_path.exists():
        raise FileNotFoundError(f"Could not find mask for case_id={case_id}: {mask_path}")

    image = load_grayscale_image(image_path)
    raw_mask = np.load(mask_path)
    channel_masks, components = extract_components(raw_mask)
    union_mask = np.clip(np.sum(channel_masks, axis=0), 0, 1).astype(np.uint8)
    component_map = make_component_map(components, union_mask.shape)
    top_pairs = build_similarity_table(image, components, margin=margin)

    overview_cols = 3
    detail_cols = 4 if len(components) > 6 else 3
    detail_rows = max(1, math.ceil(len(components) / detail_cols))
    total_rows = 1 + detail_rows

    fig = plt.figure(figsize=(4.8 * detail_cols, 4.3 * total_rows))
    grid = fig.add_gridspec(total_rows, detail_cols, height_ratios=[1.0] + [1.0] * detail_rows)

    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Image {case_id}")
    ax.axis("off")

    ax = fig.add_subplot(grid[0, 1])
    ax.imshow(apply_overlay(image, union_mask, color=(1.0, 0.2, 0.2)))
    ax.set_title(f"Union mask ({len(components)} components)")
    ax.axis("off")

    ax = fig.add_subplot(grid[0, 2])
    labeled = np.ma.masked_where(component_map == 0, component_map)
    ax.imshow(image, cmap="gray")
    ax.imshow(labeled, cmap="nipy_spectral", alpha=0.6)
    for idx, component in enumerate(components, start=1):
        y0, x0, y1, x1 = component.bbox
        ax.text(
            x0,
            y0,
            str(idx),
            color="white",
            fontsize=10,
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 1},
        )
    ax.set_title("Component labels")
    ax.axis("off")

    for idx, component in enumerate(components):
        row = 1 + idx // detail_cols
        col = idx % detail_cols
        ax = fig.add_subplot(grid[row, col])
        expanded_bbox = expand_bbox(component.bbox, image.shape, margin=margin)
        crop_img, crop_mask = crop_region(image, component.mask, expanded_bbox)
        ax.imshow(apply_overlay(crop_img, crop_mask, color=(0.1, 0.8, 1.0)))
        y0, x0, y1, x1 = component.bbox
        ax.set_title(
            f"#{idx + 1} ch={component.channel_idx + 1} area={component.area}\n"
            f"bbox=({y0}:{y1}, {x0}:{x1})"
        )
        ax.axis("off")

    similarity_text = "No pairwise similarity suggestions."
    if top_pairs:
        parts = []
        for similarity, i, j in top_pairs:
            parts.append(f"#{i + 1}<->#{j + 1}: {similarity:.3f}")
        similarity_text = "Top heuristic patch similarities: " + ", ".join(parts)

    fig.suptitle(
        f"{case_id} | raw mask shape={tuple(raw_mask.shape)} | channels={len(channel_masks)}\n{similarity_text}",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case_id}_components.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def auto_select_case_ids(mask_dir: Path, min_components: int, limit: int) -> list[str]:
    selected: list[str] = []
    for mask_path in sorted(mask_dir.glob("*.npy")):
        raw_mask = np.load(mask_path)
        _, components = extract_components(raw_mask)
        if len(components) >= min_components:
            selected.append(mask_path.stem)
        if len(selected) >= limit:
            break
    return selected


def main() -> None:
    args = parse_args()
    case_ids = args.case_ids or auto_select_case_ids(
        mask_dir=args.mask_dir,
        min_components=args.min_components,
        limit=args.limit,
    )
    if not case_ids:
        raise SystemExit("No case IDs matched the requested filters.")

    print(f"Rendering {len(case_ids)} case(s) from {args.mask_dir}")
    for case_id in case_ids:
        output_path = render_case(
            case_id=case_id,
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            output_dir=args.output_dir,
            margin=args.margin,
        )
        print(output_path)


if __name__ == "__main__":
    main()
