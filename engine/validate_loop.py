import numpy as np
import torch
from tqdm import tqdm

from inference.postprocess import post_process_prediction
from recodai_f1 import calculate_f1_score


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    sliding_window_fn,
    pixel_util,
    pred_threshold: float,
    harden_temperature: float,
    hard_clip_low: float,
    hard_clip_high: float,
    min_component_area: int,
    epoch_idx: int,
) -> float:
    model.eval()
    f1s = []

    for imgs, masks in tqdm(val_loader, desc=f"epoch {epoch_idx + 1} val"):
        imgs = imgs.to(device)
        masks = masks.to(device)

        for i in range(imgs.shape[0]):
            img = imgs[i]
            mask = masks[i]

            logits = sliding_window_fn(img, model, device)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs = post_process_prediction(
                probs=probs,
                pixel_util=pixel_util,
                threshold=pred_threshold,
                harden_temperature=harden_temperature,
                hard_clip_low=hard_clip_low,
                hard_clip_high=hard_clip_high,
                min_component_area=min_component_area,
            )
            gt = mask.cpu().numpy()[0]

            pred_bin = (probs >= pred_threshold).astype(np.uint8)
            gt_bin = (gt >= 0.5).astype(np.uint8)
            f1s.append(calculate_f1_score(pred_bin, gt_bin))

    return float(np.mean(f1s))
