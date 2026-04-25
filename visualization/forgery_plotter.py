from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image
from scipy import ndimage

from dataset_utils import SampleRecord, find_mask_paths, load_instance_masks, load_union_mask_from_paths


@dataclass(frozen=True, slots=True)
class ForgeryCase:
    case_id: str
    forged_path: Path
    authentic_path: Path | None
    mask_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class SourceTargetMasks:
    union_mask: np.ndarray
    source_mask: np.ndarray
    target_mask: np.ndarray
    diff: np.ndarray
    diff_mask: np.ndarray
    component_scores: pd.DataFrame


class ForgeryDataPlotter:
    """Matplotlib helper for exploring authentic/forged image pairs and masks."""

    def __init__(self, data_root: str | Path = "data", mask_alpha: float = 0.45, cmap: str = "Reds"):
        self.data_root = Path(data_root)
        self.mask_alpha = mask_alpha
        self.cmap = cmap

    @property
    def forged_dir(self) -> Path:
        return self.data_root / "train_images" / "forged"

    @property
    def authentic_dir(self) -> Path:
        return self.data_root / "train_images" / "authentic"

    @property
    def mask_dir(self) -> Path:
        return self.data_root / "train_masks"

    def list_case_ids(self, require_authentic: bool = False, require_mask: bool = True) -> list[str]:
        """Return sorted forged case ids that are available for visualization."""
        case_ids: list[str] = []
        for path in sorted(self.forged_dir.glob("*.png")):
            case_id = path.stem
            if require_authentic and not (self.authentic_dir / f"{case_id}.png").exists():
                continue
            if require_mask and not find_mask_paths(case_id, data_root=self.data_root):
                continue
            case_ids.append(case_id)
        return case_ids

    def case_from_id(self, case_id: str) -> ForgeryCase:
        forged_path = self.forged_dir / f"{case_id}.png"
        if not forged_path.exists():
            raise FileNotFoundError(f"Forged image not found: {forged_path}")

        authentic_path = self.authentic_dir / f"{case_id}.png"
        mask_paths = tuple(find_mask_paths(case_id, data_root=self.data_root))
        return ForgeryCase(
            case_id=case_id,
            forged_path=forged_path,
            authentic_path=authentic_path if authentic_path.exists() else None,
            mask_paths=mask_paths,
        )

    def case_from_sample(self, sample: SampleRecord) -> ForgeryCase:
        authentic_path = self.authentic_dir / f"{sample.case_id}.png"
        return ForgeryCase(
            case_id=sample.case_id,
            forged_path=sample.image_path,
            authentic_path=authentic_path if authentic_path.exists() else None,
            mask_paths=tuple(sample.mask_paths),
        )

    def load_image(self, path: str | Path) -> np.ndarray:
        image = np.asarray(Image.open(path))
        if image.ndim == 2:
            return image
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, case: str | ForgeryCase | SampleRecord) -> np.ndarray:
        resolved = self._resolve_case(case)
        if not resolved.mask_paths:
            raise FileNotFoundError(f"No masks found for case_id={resolved.case_id}")
        return load_union_mask_from_paths(resolved.mask_paths)

    def load_instances(self, case: str | ForgeryCase | SampleRecord) -> list[np.ndarray]:
        resolved = self._resolve_case(case)
        if not resolved.mask_paths:
            return []
        return load_instance_masks(resolved.mask_paths)

    def summarize_cases(self, case_ids: Iterable[str] | None = None) -> pd.DataFrame:
        """Build a small table with image sizes and mask coverage for quick EDA."""
        rows = []
        for case_id in case_ids or self.list_case_ids():
            case = self.case_from_id(case_id)
            forged = self.load_image(case.forged_path)
            mask = self.load_mask(case)
            rows.append(
                {
                    "case_id": case.case_id,
                    "height": int(forged.shape[0]),
                    "width": int(forged.shape[1]),
                    "channels": 1 if forged.ndim == 2 else int(forged.shape[2]),
                    "mask_pixels": int(mask.sum()),
                    "mask_fraction": float(mask.mean()),
                    "instance_count": len(self.load_instances(case)),
                    "has_authentic_pair": case.authentic_path is not None,
                }
            )
        return pd.DataFrame(rows)

    def derive_source_target_masks(
        self,
        case: str | ForgeryCase | SampleRecord,
        *,
        diff_threshold: float = 5.0,
        split_strategy: Literal["component", "pixel"] = "component",
        component_change_fraction: float = 0.25,
        min_component_area: int = 0,
    ) -> SourceTargetMasks:
        """Split the union copy-move mask into source and pasted-target masks.

        By default, the diff is only used to classify whole connected components
        from the clean GT union mask. Low-intensity differences are ignored so
        faint gray source-region noise does not dominate the decision.
        """
        resolved = self._resolve_case(case)
        if resolved.authentic_path is None:
            raise FileNotFoundError(f"No authentic pair found for case_id={resolved.case_id}")
        if split_strategy not in {"component", "pixel"}:
            raise ValueError(f"split_strategy must be 'component' or 'pixel', got {split_strategy!r}")

        authentic = self.load_image(resolved.authentic_path)
        forged = self.load_image(resolved.forged_path)
        union_mask = self.load_mask(resolved) > 0
        diff = self._abs_difference(authentic, forged)
        diff_mask = diff > diff_threshold

        if split_strategy == "pixel":
            target_mask = union_mask & diff_mask
            source_mask = union_mask & ~target_mask
            component_scores = pd.DataFrame()
        else:
            source_mask, target_mask, component_scores = self._split_union_components_by_diff(
                union_mask,
                diff,
                diff_mask,
                component_change_fraction=component_change_fraction,
                min_component_area=min_component_area,
            )

        return SourceTargetMasks(
            union_mask=union_mask.astype(np.uint8),
            source_mask=source_mask.astype(np.uint8),
            target_mask=target_mask.astype(np.uint8),
            diff=diff,
            diff_mask=diff_mask.astype(np.uint8),
            component_scores=component_scores,
        )

    def plot_case(
        self,
        case: str | ForgeryCase | SampleRecord,
        *,
        show_difference: bool = True,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, np.ndarray]:
        """Plot authentic image, forged image, forged image with mask, and optional absolute difference."""
        resolved = self._resolve_case(case)
        forged = self.load_image(resolved.forged_path)
        authentic = self.load_image(resolved.authentic_path) if resolved.authentic_path else None
        mask = self.load_mask(resolved) if resolved.mask_paths else np.zeros(forged.shape[:2], dtype=np.uint8)

        panel_count = 4 if show_difference and authentic is not None else 3
        figsize = figsize or (4.8 * panel_count, 4.8)
        fig, axes = plt.subplots(1, panel_count, figsize=figsize)
        axes = np.asarray(axes).reshape(-1)

        if authentic is None:
            self._show_image(axes[0], forged, "Forged")
        else:
            self._show_image(axes[0], authentic, "Authentic")
            self._show_image(axes[1], forged, "Forged")

        overlay_ax = axes[2 if authentic is not None else 1]
        self._show_image(overlay_ax, forged, "Forged + mask")
        self._show_mask_overlay(overlay_ax, mask)

        mask_ax = axes[2] if authentic is None else axes[3] if panel_count == 4 else None
        if mask_ax is not None and (authentic is None or not show_difference):
            self._show_mask(mask_ax, mask, "Mask")

        if show_difference and authentic is not None:
            diff = self._abs_difference(authentic, forged)
            diff_ax = axes[3]
            diff_ax.imshow(diff, cmap="magma")
            diff_ax.set_title("Absolute difference")
            diff_ax.axis("off")
            self._show_mask_overlay(diff_ax, mask, alpha=0.25)

        fig.suptitle(f"Case {resolved.case_id}", y=1.02)
        fig.tight_layout()
        return fig, axes

    def plot_source_target_split(
        self,
        case: str | ForgeryCase | SampleRecord,
        *,
        diff_threshold: float = 5.0,
        split_strategy: Literal["component", "pixel"] = "component",
        component_change_fraction: float = 0.25,
        min_component_area: int = 0,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, np.ndarray]:
        """Plot the derived BusterNet-style source/target mask split."""
        resolved = self._resolve_case(case)
        masks = self.derive_source_target_masks(
            resolved,
            diff_threshold=diff_threshold,
            split_strategy=split_strategy,
            component_change_fraction=component_change_fraction,
            min_component_area=min_component_area,
        )
        forged = self.load_image(resolved.forged_path)
        authentic = self.load_image(resolved.authentic_path) if resolved.authentic_path else None

        figsize = figsize or (24, 4.8)
        fig, axes = plt.subplots(1, 5, figsize=figsize)
        axes = np.asarray(axes).reshape(-1)

        self._show_image(axes[0], authentic, "Authentic")
        self._show_image(axes[1], forged, "Forged")
        self._show_mask(axes[2], masks.union_mask, "GT union")
        self._show_mask(axes[3], masks.target_mask, "Target components")
        self._show_mask(axes[4], masks.source_mask, "Source components")

        fig.suptitle(f"Source/target split for case {resolved.case_id}", y=1.02)
        fig.tight_layout()
        return fig, axes

    def plot_instances(
        self,
        case: str | ForgeryCase | SampleRecord,
        *,
        cols: int = 4,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, np.ndarray]:
        resolved = self._resolve_case(case)
        instances = self.load_instances(resolved)
        if not instances:
            raise FileNotFoundError(f"No masks found for case_id={resolved.case_id}")

        cols = min(cols, len(instances))
        rows = math.ceil(len(instances) / cols)
        figsize = figsize or (3.5 * cols, 3.5 * rows)
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.asarray(axes).reshape(-1)

        for idx, ax in enumerate(axes):
            if idx < len(instances):
                self._show_mask(ax, instances[idx], f"Instance {idx}")
            else:
                ax.axis("off")

        fig.suptitle(f"Mask instances for case {resolved.case_id}", y=1.02)
        fig.tight_layout()
        return fig, axes

    def plot_random_cases(
        self,
        n: int = 6,
        *,
        seed: int | None = 0,
        cols: int = 3,
    ) -> tuple[Figure, np.ndarray]:
        case_ids = self.list_case_ids(require_authentic=True, require_mask=True)
        if not case_ids:
            raise FileNotFoundError("No forged cases with authentic pairs and masks were found.")

        rng = random.Random(seed)
        selected = rng.sample(case_ids, k=min(n, len(case_ids)))
        rows = math.ceil(len(selected) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.8 * rows))
        axes = np.asarray(axes).reshape(-1)

        for idx, ax in enumerate(axes):
            if idx >= len(selected):
                ax.axis("off")
                continue
            case = self.case_from_id(selected[idx])
            forged = self.load_image(case.forged_path)
            mask = self.load_mask(case)
            self._show_image(ax, forged, f"Case {case.case_id}")
            self._show_mask_overlay(ax, mask)

        fig.tight_layout()
        return fig, axes

    def _resolve_case(self, case: str | ForgeryCase | SampleRecord) -> ForgeryCase:
        if isinstance(case, ForgeryCase):
            return case
        if isinstance(case, SampleRecord):
            return self.case_from_sample(case)
        return self.case_from_id(str(case))

    def _show_image(self, ax: Axes, image: np.ndarray, title: str) -> None:
        if image.ndim == 2:
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    def _show_mask(self, ax: Axes, mask: np.ndarray, title: str) -> None:
        ax.imshow(mask > 0, cmap="gray", interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")

    def _show_mask_overlay(self, ax: Axes, mask: np.ndarray, alpha: float | None = None) -> None:
        masked = np.ma.masked_where(mask <= 0, mask)
        ax.imshow(masked, cmap=self.cmap, alpha=self.mask_alpha if alpha is None else alpha, interpolation="nearest")

    def _split_union_components_by_diff(
        self,
        union_mask: np.ndarray,
        diff: np.ndarray,
        diff_mask: np.ndarray,
        *,
        component_change_fraction: float,
        min_component_area: int,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        labels, label_count = ndimage.label(union_mask, structure=np.ones((3, 3), dtype=np.uint8))
        target_mask = np.zeros_like(union_mask, dtype=bool)
        source_mask = np.zeros_like(union_mask, dtype=bool)
        rows = []

        for label_idx in range(1, label_count + 1):
            component = labels == label_idx
            area = int(component.sum())
            changed_pixels = int(diff_mask[component].sum())
            changed_fraction = changed_pixels / area if area else 0.0
            mean_diff = float(diff[component].mean()) if area else 0.0
            max_diff = float(diff[component].max()) if area else 0.0
            is_tiny = area < min_component_area
            is_target = (not is_tiny) and changed_fraction >= component_change_fraction

            rows.append(
                {
                    "component": label_idx,
                    "area": area,
                    "changed_pixels": changed_pixels,
                    "changed_fraction": changed_fraction,
                    "mean_diff": mean_diff,
                    "max_diff": max_diff,
                    "is_tiny": is_tiny,
                    "is_target": is_target,
                }
            )

            if is_tiny:
                continue
            if is_target:
                target_mask |= component
            else:
                source_mask |= component

        component_scores = pd.DataFrame(rows)
        if target_mask.any() or component_scores.empty:
            return source_mask, target_mask, component_scores

        candidates = component_scores[~component_scores["is_tiny"] & (component_scores["changed_pixels"] > 0)]
        if candidates.empty:
            return source_mask, target_mask, component_scores

        fallback_component = int(candidates.sort_values(["changed_fraction", "mean_diff"], ascending=False).iloc[0]["component"])
        fallback_mask = labels == fallback_component
        target_mask |= fallback_mask
        source_mask &= ~fallback_mask
        component_scores.loc[component_scores["component"] == fallback_component, "is_target"] = True
        return source_mask, target_mask, component_scores

    def _abs_difference(self, authentic: np.ndarray, forged: np.ndarray) -> np.ndarray:
        authentic = self._as_float_gray(authentic)
        forged = self._as_float_gray(forged)
        if authentic.shape != forged.shape:
            raise ValueError(f"Authentic and forged images have different shapes: {authentic.shape} vs {forged.shape}")
        return np.abs(forged - authentic)

    def _as_float_gray(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        if image.ndim == 3:
            image = image.mean(axis=2)
        return image
