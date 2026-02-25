import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset_utils import load_image, load_union_mask


class ForgeryDataset(Dataset):
    def __init__(
        self,
        case_ids,
        target_size: int = 256,
        use_rgb: bool = False,
        normalize_rgb: bool = False,
        rgb_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        rgb_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.case_ids = case_ids
        self.target_size = target_size
        self.use_rgb = use_rgb
        self.normalize_rgb = normalize_rgb
        self.rgb_mean = np.asarray(rgb_mean, dtype=np.float32).reshape(1, 1, 3)
        self.rgb_std = np.asarray(rgb_std, dtype=np.float32).reshape(1, 1, 3)

    def __len__(self) -> int:
        return len(self.case_ids)

    def _resize(self, arr: np.ndarray, size: int, is_mask: bool = False) -> np.ndarray:
        arr = np.squeeze(arr)

        if arr.ndim == 1:
            arr = arr.reshape((1, -1))

        if (not self.use_rgb) and arr.ndim == 3:
            arr = arr.mean(axis=2)

        img = Image.fromarray(arr.astype(np.uint8))
        if is_mask:
            img = img.resize((size, size), resample=Image.NEAREST)
        else:
            img = img.resize((size, size), resample=Image.BILINEAR)
        arr = np.array(img)

        if self.use_rgb and not is_mask:
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            elif arr.ndim == 3 and arr.shape[2] != 3:
                arr = np.repeat(arr[..., :1], 3, axis=2)

        return arr

    def __getitem__(self, idx: int):
        cid = self.case_ids[idx]
        img = load_image(cid)
        mask = load_union_mask(cid)
        mask = np.asarray(mask)

        if mask.ndim == 3:
            mask = (mask.max(axis=0) > 0).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

        img = self._resize(img, self.target_size, is_mask=False)
        mask = self._resize(mask * 255, self.target_size, is_mask=True)

        img = (img / 255.0).astype(np.float32)
        if self.use_rgb and self.normalize_rgb:
            img = (img - self.rgb_mean) / self.rgb_std
        mask = (mask > 0).astype(np.float32)

        if self.use_rgb:
            img = torch.from_numpy(img).permute(2, 0, 1)
        else:
            img = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask
