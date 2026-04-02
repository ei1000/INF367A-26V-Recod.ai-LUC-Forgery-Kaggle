from pathlib import Path
from enum import Enum
import numpy as np
import torch
from torch.utils.data import Dataset
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

def dino_transform(size):
    to_tensor = v2.ToImage()
    resize = v2.Resize((size, size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

def imagenet_transform(size):
    to_tensor = v2.ToImage()
    resize = v2.Resize((size, size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

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
