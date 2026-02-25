import numpy as np
import torch
import torch.nn.functional as F

PATCH_SIZE = 256 # DINO-dependent
BATCH_SIZE = 32
EPS = 1e-5

STRIDE = PATCH_SIZE // 2 # can be toggled

def gaussian_weight(patch_size, sigma=0.125):
    ax = np.linspace(-1, 1, patch_size)
    xx, yy = np.meshgrid(ax, ax)
    dist = np.sqrt(xx**2 + yy**2)
    weight = np.exp(-(dist**2) / (2 * sigma**2))
    return weight


'''
Collects all crops,
model processes in parallell as patch
'''
def predict_batched_crops(crops, model, device):
    # crops = list of tensors (1,H,W)
    batch = torch.stack(crops, dim=0)
    batch = batch.to(device)  # Ensure batch is on correct device
    preds = model(batch)              
    return preds


'''
Runs sliding window inference on image.
'''
def sliding_window(img, model, device):
    H, W = img.squeeze(0).shape

    # small image case
    if H < PATCH_SIZE or W < PATCH_SIZE:
        patch = F.pad(img, (0, PATCH_SIZE-W, 0, PATCH_SIZE-H))
        return model(patch[None].to(device))[0]

    weight = torch.from_numpy(gaussian_weight(PATCH_SIZE)).to(device)

    prob_map = torch.zeros((H, W), device=device)
    weight_map = torch.zeros((H, W), device=device) + EPS

    crops, coords = [], []

    # collect all crops
    for y in range(0, H, STRIDE):
        for x in range(0, W, STRIDE):
            crop = img[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            pad_h = max(0, PATCH_SIZE - crop.shape[1])
            pad_w = max(0, PATCH_SIZE - crop.shape[2])
            crop = F.pad(crop, (0, pad_w, 0, pad_h), mode="constant")

            crops.append(crop)
            coords.append((y, x))

    # batched inference
    model.eval()
    with torch.no_grad():
        for i in range(0, len(crops), BATCH_SIZE):
            batch_crops = crops[i:i+BATCH_SIZE]
            batch_coords = coords[i:i+BATCH_SIZE]

            preds = predict_batched_crops(batch_crops, model, device)  

            for pred, (y, x) in zip(preds, batch_coords):
                pred = pred.squeeze(0) 

                h = min(PATCH_SIZE, H - y)
                w = min(PATCH_SIZE, W - x)

                prob_map[y:y+h, x:x+w] += pred[:h, :w] * weight[:h, :w]
                weight_map[y:y+h, x:x+w] += weight[:h, :w]

    return prob_map / weight_map
