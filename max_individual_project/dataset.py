from pathlib import Path
from enum import Enum
import math
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import v2, InterpolationMode
from PIL import Image

def resolve_data_root() -> Path:
    root = Path("data")
    if root.exists():
        return root
    alt = Path(__file__).resolve().parent.parent / "data"
    if alt.exists():
        return alt
    raise FileNotFoundError("Could not find data directory. Checked ./data and ../data.")


def imagenet_rgb_transform(size):
    to_tensor = v2.ToImage()
    resize = v2.Resize((size, size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])


def dino_transform(size):
    return imagenet_rgb_transform(size)


def imagenet_transform(size):
    return imagenet_rgb_transform(size)


def imagenet_normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std

def regular_transform(size):
    to_tensor = v2.ToImage()
    resize = v2.Resize((size, size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)

    return v2.Compose([to_tensor, resize, to_float])


def normalize_mask_array(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return (mask > 0).astype(np.uint8)

    if mask.ndim == 3:
        # Dataset masks can be stored as channel-first or channel-last, with a
        # variable number of instance/source channels. For binary supervision we
        # only need "any forged pixel here?", so collapse the channel axis.
        if mask.shape[0] <= 16 and mask.shape[0] <= mask.shape[-1] and mask.shape[0] <= mask.shape[-2]:
            mask = mask.max(axis=0)
        elif mask.shape[-1] <= 16 and mask.shape[-1] <= mask.shape[0] and mask.shape[-1] <= mask.shape[1]:
            mask = mask.max(axis=-1)
        else:
            raise ValueError(f"Could not infer channel axis for mask with shape {mask.shape}")
        return (mask > 0).astype(np.uint8)

    raise ValueError(f"Unsupported mask shape {mask.shape}")


def resolve_image_transform(
    feature_backbone: str,
    use_dino_transform: bool,
    cnn_backbone: str,
    separate_transforms: bool,
):
    if separate_transforms:
        return regular_transform
    if feature_backbone in ("dino", "dino_single") and use_dino_transform:
        return dino_transform
    if feature_backbone == "cnn" and cnn_backbone == "pretrained":
        return imagenet_transform
    return regular_transform


def combine_datasets(dataset_list):
    if not dataset_list:
        return None
    if len(dataset_list) == 1:
        return dataset_list[0]
    return ConcatDataset(dataset_list)


def _allocate_split_counts(total_count: int, validation_split: float, test_split: float) -> tuple[int, int, int]:
    if total_count < 0:
        raise ValueError(f"total_count must be non-negative, got {total_count}")
    if validation_split < 0.0 or test_split < 0.0:
        raise ValueError(
            f"validation_split and test_split must be non-negative, got {validation_split} and {test_split}"
        )
    if (validation_split + test_split) >= 1.0:
        raise ValueError(
            f"validation_split + test_split must be < 1.0, got {validation_split + test_split:.4f}"
        )
    if total_count == 0:
        return 0, 0, 0
    if validation_split <= 0.0 and test_split <= 0.0:
        return total_count, 0, 0

    train_split = 1.0 - validation_split - test_split
    split_targets = {
        "train": total_count * train_split,
        "val": total_count * validation_split,
        "test": total_count * test_split,
    }
    split_counts = {name: int(math.floor(target)) for name, target in split_targets.items()}
    remainder = total_count - sum(split_counts.values())
    if remainder > 0:
        for name, _ in sorted(
            split_targets.items(),
            key=lambda item: item[1] - math.floor(item[1]),
            reverse=True,
        ):
            if remainder <= 0:
                break
            split_counts[name] += 1
            remainder -= 1

    required_min = {
        "train": 1,
        "val": 1 if validation_split > 0.0 else 0,
        "test": 1 if test_split > 0.0 else 0,
    }
    required_total = sum(required_min.values())
    if total_count < required_total:
        # Not enough examples to honor all three non-empty splits for this label. Keep
        # at least one training sample and allocate the remainder to validation/test by ratio.
        split_counts = {"train": 1, "val": 0, "test": 0}
        remaining = total_count - 1
        if remaining > 0 and validation_split > 0.0:
            split_counts["val"] = 1
            remaining -= 1
        if remaining > 0 and test_split > 0.0:
            split_counts["test"] = 1
            remaining -= 1
        while remaining > 0:
            if validation_split >= test_split:
                split_counts["val"] += 1
            else:
                split_counts["test"] += 1
            remaining -= 1
        return split_counts["train"], split_counts["val"], split_counts["test"]

    for name, minimum in required_min.items():
        while split_counts[name] < minimum:
            donor = max(
                (candidate for candidate in split_counts if split_counts[candidate] > required_min[candidate]),
                key=lambda candidate: split_counts[candidate],
            )
            split_counts[donor] -= 1
            split_counts[name] += 1

    return split_counts["train"], split_counts["val"], split_counts["test"]


def split_indices_by_label_three_way(
    samples,
    validation_split: float,
    test_split: float,
    seed: int,
):
    if validation_split <= 0.0 and test_split <= 0.0:
        indices = list(range(len(samples)))
        return indices, [], []

    label_to_indices = {}
    for idx, (_, label) in enumerate(samples):
        label_to_indices.setdefault(label, []).append(idx)

    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    for indices in label_to_indices.values():
        shuffled = [indices[i] for i in torch.randperm(len(indices), generator=generator).tolist()]
        train_count, val_count, test_count = _allocate_split_counts(
            len(shuffled),
            validation_split=validation_split,
            test_split=test_split,
        )

        val_end = val_count
        test_end = val_end + test_count
        val_indices.extend(shuffled[:val_end])
        test_indices.extend(shuffled[val_end:test_end])
        train_indices.extend(shuffled[test_end:])

        if len(shuffled[test_end:]) != train_count:
            raise RuntimeError("Split allocation mismatch while building train/val/test partitions.")

    train_indices.sort()
    val_indices.sort()
    test_indices.sort()
    return train_indices, val_indices, test_indices


def split_indices_by_label(samples, validation_split: float, seed: int):
    train_indices, val_indices, _ = split_indices_by_label_three_way(
        samples,
        validation_split=validation_split,
        test_split=0.0,
        seed=seed,
    )
    return train_indices, val_indices

class ForgeryDataset(Dataset):
    def __init__(self, samples, mask_dir: Path | None=None, size=384, transform=dino_transform, return_path: bool = False):
        self.samples = samples
        self.mask_dir = mask_dir
        self.size = size
        self.return_path = return_path


        self.image_transforms = transform(size)

        self.mask_transforms = v2.Compose([
            v2.Resize((size, size), 
            interpolation=InterpolationMode.NEAREST),
        ])
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.image_transforms(image)

        if self.mask_dir is not None and "forged" in img_path.parent.name:
            mask_path = self.mask_dir / img_path.name.replace(".png", ".npy")
            mask = np.load(mask_path)
            mask = normalize_mask_array(mask)

            mask = torch.from_numpy(mask)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = self.mask_transforms(mask)
            mask = mask.squeeze(0).long()
        else:
            mask = torch.zeros((self.size, self.size), dtype=torch.long)

        if self.return_path:
            return image, mask, label, str(img_path)

        return image, mask, label


class Datasets(Enum):
    TRAIN = [{'images': 'train_images', 'masks': 'train_masks'}]
    SUPPLEMENT = [{'images': 'supplemental_images', 'masks': 'supplemental_masks'}]
    CASIA = [{'images': 'casia_cmfd_images', 'masks': 'casia_cmfd_masks_np'}]
    TEST = [{'images': 'test_images', 'masks': None}]
    SELF_PROCURED = [{'images': 'self_procured', 'masks': None}]

    ALL_TRAIN = TRAIN + CASIA + SUPPLEMENT
    KAGGLE_TRAIN = TRAIN + SUPPLEMENT
