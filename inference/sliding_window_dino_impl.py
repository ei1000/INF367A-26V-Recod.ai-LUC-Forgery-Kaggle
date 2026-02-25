import numpy as np
import torch
import torch.nn.functional as F

PATCH_SIZE = 256
BATCH_SIZE = 32
EPS = 1e-5
STRIDE = PATCH_SIZE // 2


def gaussian_weight(patch_size, sigma=0.125):
    ax = np.linspace(-1, 1, patch_size)
    xx, yy = np.meshgrid(ax, ax)
    dist = np.sqrt(xx**2 + yy**2)
    return np.exp(-(dist**2) / (2 * sigma**2))


def predict_batched_crops(crops, model, device):
    batch = torch.stack(crops, dim=0).to(device)
    return model(batch)


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

    h_img, w_img = img.shape[-2], img.shape[-1]

    if h_img < patch_size or w_img < patch_size:
        patch = F.pad(img, (0, patch_size - w_img, 0, patch_size - h_img))
        return model(patch[None].to(device))[0]

    weight = torch.from_numpy(gaussian_weight(patch_size)).to(device=device, dtype=torch.float32)
    prob_map = torch.zeros((h_img, w_img), device=device)
    weight_map = torch.zeros((h_img, w_img), device=device) + EPS

    crops, coords = [], []
    for y in range(0, h_img, stride):
        for x in range(0, w_img, stride):
            crop = img[:, y:y + patch_size, x:x + patch_size]
            pad_h = max(0, patch_size - crop.shape[1])
            pad_w = max(0, patch_size - crop.shape[2])
            crop = F.pad(crop, (0, pad_w, 0, pad_h), mode="constant")
            crops.append(crop)
            coords.append((y, x))

    model.eval()
    with torch.no_grad():
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
