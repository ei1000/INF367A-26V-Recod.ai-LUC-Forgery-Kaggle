from pathlib import Path
from enum import Enum
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2, InterpolationMode
from PIL import Image

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

class ForgeryDataset(Dataset):
    def __init__(self, samples, mask_dir: Path | None=None, size=384, transform=dino_transform):
        self.samples = samples
        self.mask_dir = mask_dir
        self.size = size


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

            # Normalize masks to a single-channel 2D label map.
            if mask.ndim == 3:
                # Handle both channel-first and channel-last encodings.
                if mask.shape[0] in (1, 2, 3):
                    mask = mask[0] if mask.shape[0] == 1 else mask.argmax(axis=0)
                else:
                    mask = mask[..., 0] if mask.shape[-1] == 1 else mask.argmax(axis=-1)

            mask = torch.from_numpy(mask)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = self.mask_transforms(mask)
            mask = mask.squeeze(0).long()
        else:
            mask = torch.from_numpy(np.zeros((self.size,  self.size)))

        return image, mask, label


class Datasets(Enum):
    TRAIN = [{'images': 'train_images', 'masks': 'train_masks'}]
    SUPPLEMENT = [{'images': 'supplemental_images', 'masks': 'supplemental_masks'}]
    TEST = [{'images': 'test_images', 'masks': None}]
    SELF_PROCURED = [{'images': 'self_procured', 'masks': None}]

    ALL_TRAIN = TRAIN + SUPPLEMENT
