from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from PIL import Image

DATA = Path("data")
LabelName = Literal["forged", "authentic"]
SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class SampleRecord:
    sample_id: str
    case_id: str
    label: LabelName
    image_path: Path
    mask_paths: tuple[Path, ...]
    group_id: str
    split: SplitName | None = None

    def with_split(self, split: SplitName) -> "SampleRecord":
        return replace(self, split=split)


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Convert RGB-like images to grayscale and leave grayscale images unchanged."""
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img


def _validate_label(label: str) -> LabelName:
    if label not in {"forged", "authentic"}:
        raise ValueError(f"label must be 'forged' or 'authentic', got {label!r}")
    return label  # type: ignore[return-value]


def find_image_path(case_id: str, label: LabelName | None = None, data_root: Path = DATA) -> Path:
    """Find an image path for a case ID, optionally constrained to one label directory."""
    data_root = Path(data_root)
    if label is not None:
        label = _validate_label(label)
        candidates = [data_root / "train_images" / label / f"{case_id}.png"]
    else:
        candidates = [
            data_root / "train_images" / "forged" / f"{case_id}.png",
            data_root / "train_images" / "authentic" / f"{case_id}.png",
            data_root / "train_images" / f"{case_id}.png",
        ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Image not found for case_id={case_id}. Tried: {candidates}")


def load_image_from_path(path: Path) -> np.ndarray:
    """Load an exact image path as a float32 grayscale numpy array."""
    img = np.array(Image.open(path))
    return _to_gray(img).astype(np.float32)


def load_image(case_id: str) -> np.ndarray:
    """Legacy convenience loader that returns an image as a grayscale numpy array."""
    p = find_image_path(case_id)
    return load_image_from_path(p)


def find_mask_paths(case_id: str, data_root: Path = DATA) -> list[Path]:
    """List deterministic mask paths for names such as 10.npy and 10_0.npy."""
    mask_dir = Path(data_root) / "train_masks"
    masks = sorted(mask_dir.glob(f"{case_id}.npy")) + sorted(mask_dir.glob(f"{case_id}_*.npy"))
    return masks


def _load_binary_mask(path: Path) -> np.ndarray:
    mask = np.load(path)
    mask = np.squeeze(mask)
    return (mask > 0).astype(np.uint8)


def _coerce_mask_paths(mask_paths_or_case_id: str | Path | Iterable[Path]) -> tuple[Path, ...]:
    if isinstance(mask_paths_or_case_id, str):
        return tuple(find_mask_paths(mask_paths_or_case_id))
    if isinstance(mask_paths_or_case_id, Path):
        return (mask_paths_or_case_id,)
    return tuple(Path(path) for path in mask_paths_or_case_id)


def load_instance_masks(mask_paths_or_case_id: str | Path | Iterable[Path]) -> list[np.ndarray]:
    """Load individual binary instance masks from mask paths or a legacy case ID."""
    mask_paths = _coerce_mask_paths(mask_paths_or_case_id)
    return [_load_binary_mask(path) for path in mask_paths]


def load_union_mask_from_paths(mask_paths: Iterable[Path]) -> np.ndarray:
    """Load multiple instance masks and return their binary union."""
    instances = load_instance_masks(mask_paths)
    if len(instances) == 0:
        raise FileNotFoundError("No masks found")

    union = np.zeros_like(instances[0], dtype=np.uint8)
    for mask in instances:
        union = np.maximum(union, mask.astype(np.uint8))
    return union.astype(np.uint8, copy=False)


def load_union_mask(case_id: str) -> np.ndarray:
    """Legacy union-mask loader for a forged case ID."""
    mask_paths = find_mask_paths(case_id)
    if len(mask_paths) == 0:
        raise FileNotFoundError(f"No masks found for case_id={case_id}")

    return load_union_mask_from_paths(mask_paths)


def list_labeled_samples(data_root: Path = DATA) -> list[SampleRecord]:
    """Discover all labeled forged and authentic training samples."""
    data_root = Path(data_root)
    samples: list[SampleRecord] = []

    for label in ("authentic", "forged"):
        image_dir = data_root / "train_images" / label
        for image_path in sorted(image_dir.glob("*.png")):
            case_id = image_path.stem
            if label == "forged":
                mask_paths = tuple(find_mask_paths(case_id, data_root=data_root))
                if len(mask_paths) == 0:
                    raise FileNotFoundError(f"No masks found for forged case_id={case_id}")
            else:
                mask_paths = tuple()

            samples.append(
                SampleRecord(
                    sample_id=f"{label}:{case_id}",
                    case_id=case_id,
                    label=label,
                    image_path=image_path,
                    mask_paths=mask_paths,
                    group_id=case_id,
                    split=None,
                )
            )

    return sorted(samples, key=lambda sample: sample.sample_id)
