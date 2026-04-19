from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

PATCH_SIZE = 256
BATCH_SIZE = 32
EPS = 1e-5
STRIDE = PATCH_SIZE // 2


def compute_window_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if length <= patch_size:
        return [0]

    starts = list(range(0, max(1, length - patch_size + 1), stride))
    final_start = length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return sorted(set(starts))


@lru_cache(maxsize=16)
def gaussian_weight_numpy(patch_size: int, sigma: float = 0.125) -> np.ndarray:
    ax = np.linspace(-1, 1, patch_size)
    xx, yy = np.meshgrid(ax, ax)
    dist = np.sqrt(xx**2 + yy**2)
    return np.exp(-(dist**2) / (2 * sigma**2)).astype(np.float32)


def gaussian_weight(patch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.from_numpy(gaussian_weight_numpy(patch_size)).to(device=device, dtype=dtype)


def predict_batched_crops(crops: list[torch.Tensor], model, device: torch.device) -> torch.Tensor:
    batch = torch.stack(crops, dim=0).to(device, non_blocking=True)
    return model(batch)


def _pad_crop_to_patch(crop: torch.Tensor, patch_size: int) -> torch.Tensor:
    pad_h = max(0, patch_size - crop.shape[1])
    pad_w = max(0, patch_size - crop.shape[2])
    if pad_h == 0 and pad_w == 0:
        return crop
    return F.pad(crop, (0, pad_w, 0, pad_h), mode="constant")


def sliding_window_dino(
    img,
    model,
    device,
    patch_size: int = PATCH_SIZE,
    stride: int | None = None,
    batch_size: int = BATCH_SIZE,
):
    if img.ndim != 3:
        raise ValueError(f"Expected image shape (C,H,W), got {tuple(img.shape)}")

    if stride is None:
        stride = patch_size // 2

    h_img, w_img = int(img.shape[-2]), int(img.shape[-1])
    model.eval()

    with torch.inference_mode():
        if h_img <= patch_size and w_img <= patch_size:
            patch = _pad_crop_to_patch(img, patch_size)
            pred = model(patch[None].to(device, non_blocking=True))[0].squeeze(0)
            return pred[:h_img, :w_img]

        y_starts = compute_window_starts(h_img, patch_size, stride)
        x_starts = compute_window_starts(w_img, patch_size, stride)

        weight = gaussian_weight(patch_size, device=device, dtype=torch.float32)
        prob_map = torch.zeros((h_img, w_img), device=device, dtype=torch.float32)
        weight_map = torch.zeros((h_img, w_img), device=device, dtype=torch.float32) + EPS

        crops: list[torch.Tensor] = []
        coords: list[tuple[int, int]] = []
        for y in y_starts:
            for x in x_starts:
                crop = img[:, y : y + patch_size, x : x + patch_size]
                crops.append(_pad_crop_to_patch(crop, patch_size))
                coords.append((y, x))

        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i:i + batch_size]
            batch_coords = coords[i:i + batch_size]
            preds = predict_batched_crops(batch_crops, model, device)

            for pred, (y, x) in zip(preds, batch_coords):
                pred = pred.squeeze(0)
                h = min(patch_size, h_img - y)
                w = min(patch_size, w_img - x)
                prob_map[y:y + h, x:x + w] += pred[:h, :w] * weight[:h, :w]
                weight_map[y:y + h, x:x + w] += weight[:h, :w]

    return prob_map / weight_map
