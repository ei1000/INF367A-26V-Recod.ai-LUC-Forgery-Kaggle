from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset_utils import SampleRecord, load_image_from_path


class BusterNetDataset(Dataset):
    """BusterNet dataset with integer labels: 0 background, 1 target, 2 source."""

    def __init__(
        self,
        samples: Sequence[SampleRecord],
        data_root: str | Path = "data",
        target_size: int = 448,
        use_rgb: bool = True,
        normalize_rgb: bool = True,
        rgb_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        rgb_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        metadata_path: str | Path | None = None,
        allowed_forged_statuses: tuple[str, ...] = ("derived_from_pair",),
        include_authentic: bool = True,
        authentic_policy: str = "paired_derived_only",
    ) -> None:
        self.data_root = Path(data_root)
        self.target_size = int(target_size)
        self.use_rgb = use_rgb
        self.normalize_rgb = normalize_rgb
        self.rgb_mean = np.asarray(rgb_mean, dtype=np.float32).reshape(1, 1, 3)
        self.rgb_std = np.asarray(rgb_std, dtype=np.float32).reshape(1, 1, 3)
        self.metadata_path = (
            Path(metadata_path) if metadata_path is not None else self.data_root / "train_masks_source_target_metadata.csv"
        )
        self.allowed_forged_statuses = set(allowed_forged_statuses)
        self.include_authentic = include_authentic
        self.authentic_policy = authentic_policy

        self._metadata = self._load_metadata()
        self._allowed_case_ids = self._case_ids_with_allowed_status()
        self.samples = self._filter_samples(samples)
        self._validate_kept_forged_masks()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = load_image_from_path(sample.image_path)
        label_map = self._load_label_map(sample, image.shape[:2])

        image = self._resize_image(image)
        label_map = self._resize_label(label_map)

        image = (image / 255.0).astype(np.float32)
        if self.use_rgb and self.normalize_rgb:
            image = (image - self.rgb_mean) / self.rgb_std

        if self.use_rgb:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        else:
            image_tensor = torch.from_numpy(image).unsqueeze(0)
        label_tensor = torch.from_numpy(label_map.astype(np.int64, copy=False))
        return image_tensor, label_tensor

    def _load_metadata(self) -> pd.DataFrame:
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"BusterNet source/target metadata not found: {self.metadata_path}. "
                "Run einar_busternet.generate_source_target_masks first."
            )

        metadata = pd.read_csv(self.metadata_path)
        required = {"case_id", "status"}
        missing = required - set(metadata.columns)
        if missing:
            raise ValueError(f"Metadata {self.metadata_path} is missing required columns: {sorted(missing)}")
        metadata = metadata.copy()
        metadata["case_id"] = metadata["case_id"].astype(str)
        return metadata

    def _case_ids_with_allowed_status(self) -> set[str]:
        allowed = self._metadata[self._metadata["status"].isin(self.allowed_forged_statuses)]
        return set(allowed["case_id"].astype(str))

    def _filter_samples(self, samples: Sequence[SampleRecord]) -> list[SampleRecord]:
        filtered: list[SampleRecord] = []
        for sample in samples:
            case_id = str(sample.case_id)
            if sample.label == "forged":
                if case_id in self._allowed_case_ids:
                    filtered.append(sample)
            elif sample.label == "authentic" and self.include_authentic:
                if self.authentic_policy == "paired_derived_only":
                    if case_id in self._allowed_case_ids:
                        filtered.append(sample)
                elif self.authentic_policy == "all":
                    filtered.append(sample)
                else:
                    raise ValueError(
                        "authentic_policy must be 'paired_derived_only' or 'all', "
                        f"got {self.authentic_policy!r}"
                    )
            elif sample.label not in {"forged", "authentic"}:
                raise ValueError(f"Unsupported sample label {sample.label!r} for sample {sample.sample_id}")
        return filtered

    def _validate_kept_forged_masks(self) -> None:
        for sample in self.samples:
            if sample.label != "forged":
                continue
            source_path, target_path = self._source_target_paths(sample.case_id)
            if not source_path.exists() or not target_path.exists():
                raise FileNotFoundError(
                    f"Missing BusterNet masks for forged case_id={sample.case_id}. "
                    f"Expected {source_path} and {target_path}. "
                    "Run einar_busternet.generate_source_target_masks first."
                )

    def _load_label_map(self, sample: SampleRecord, image_shape: tuple[int, int]) -> np.ndarray:
        if sample.label == "authentic":
            return np.zeros(image_shape, dtype=np.uint8)

        source_path, target_path = self._source_target_paths(sample.case_id)
        source_mask = np.load(source_path) > 0
        target_mask = np.load(target_path) > 0
        if source_mask.shape != target_mask.shape:
            raise ValueError(
                f"Source and target masks have different shapes for case_id={sample.case_id}: "
                f"{source_mask.shape} vs {target_mask.shape}"
            )
        if source_mask.shape != image_shape:
            raise ValueError(
                f"Mask and image shapes differ for case_id={sample.case_id}: "
                f"mask={source_mask.shape}, image={image_shape}"
            )

        label_map = np.zeros(source_mask.shape, dtype=np.uint8)
        label_map[target_mask] = 1
        label_map[source_mask] = 2
        return label_map

    def _source_target_paths(self, case_id: str) -> tuple[Path, Path]:
        source_path = self.data_root / "train_masks_source" / f"{case_id}.npy"
        target_path = self.data_root / "train_masks_target" / f"{case_id}.npy"
        return source_path, target_path

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = np.squeeze(image)
        if image.ndim == 1:
            image = image.reshape((1, -1))
        if (not self.use_rgb) and image.ndim == 3:
            image = image.mean(axis=2)

        pil_image = Image.fromarray(image.astype(np.uint8))
        pil_image = pil_image.resize((self.target_size, self.target_size), resample=Image.BILINEAR)
        resized = np.asarray(pil_image)

        if self.use_rgb:
            if resized.ndim == 2:
                resized = np.repeat(resized[..., None], 3, axis=2)
            elif resized.ndim == 3 and resized.shape[2] != 3:
                resized = np.repeat(resized[..., :1], 3, axis=2)

        return resized

    def _resize_label(self, label_map: np.ndarray) -> np.ndarray:
        # Nearest-neighbor keeps class ids finite and discrete after resize.
        pil_label = Image.fromarray(label_map.astype(np.uint8))
        pil_label = pil_label.resize((self.target_size, self.target_size), resample=Image.NEAREST)
        return np.asarray(pil_label, dtype=np.uint8)
