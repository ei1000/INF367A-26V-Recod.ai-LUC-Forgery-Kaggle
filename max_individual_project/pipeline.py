from pathlib import Path
import json
import time

import numpy as np
import scipy.ndimage
import scipy.optimize
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import ImageFolder

from cross_scale_patternmatch.pixel_propagator import PixelPropagator
from dataset import Datasets, ForgeryDataset, dino_transform, imagenet_transform, regular_transform, resolve_data_root
from datatypes import DLFDecoderInput
from feature_extractors.cnn_feature_extractor import PretrainedBackboneExtractor, PyramidFeatureExtractor
from feature_extractors.zernike_feature_extractor import PyramidZernikeExtractor, default_pq_list
from prediction.decoder import DLFDecoder
from prediction.multi_scale_dlf import MultiScaleDLF
from prediction.se_u_net import SEUNet, build_se_unet_input
from visualizer import display_image, display_pixel_offsets


def imagenet_normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def ensure_output_dirs(output_dir: str | Path):
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    predictions_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, checkpoints_dir, predictions_dir


def save_checkpoint(path: Path, epoch: int, dlf_decoder, se_model, optimizer, best_score: float | None):
    checkpoint = {
        "epoch": epoch,
        "dlf_decoder": dlf_decoder.state_dict(),
        "se_model": se_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "best_score": best_score,
        "best_loss": best_score,
    }
    torch.save(checkpoint, path)


def save_prediction_batch(predictions_dir: Path, epoch_idx: int, batch_idx: int, mask_preds: torch.Tensor):
    pred_path = predictions_dir / f"epoch_{epoch_idx + 1:03d}_batch_{batch_idx:05d}.pt"
    torch.save(mask_preds.detach().cpu(), pred_path)


def set_optimizer_learning_rate(optimizer, learning_rate: float):
    if optimizer is None:
        return

    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def synchronize_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def append_metrics_log(output_dir: Path, metrics: dict):
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics) + "\n")


def load_metrics_history(output_dir: Path) -> list[dict]:
    metrics_path = output_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []

    history = []
    with metrics_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                history.append(json.loads(line))
    return history


def save_metrics_plot(output_dir: Path):
    history = load_metrics_history(output_dir)
    if not history:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[pipeline] Skipping metrics plot: {exc}")
        return

    epochs = [entry["epoch"] for entry in history if "epoch" in entry]
    train_losses = [entry.get("train_loss", entry.get("train_dice_loss")) for entry in history]
    val_losses = [entry.get("val_loss", entry.get("val_dice_loss")) for entry in history]
    train_ldfm = [entry.get("train_ldfm") for entry in history]
    train_lmrd = [entry.get("train_lmrd") for entry in history]
    train_mprime_loss = [entry.get("train_mprime_loss") for entry in history]
    val_ldfm = [entry.get("val_ldfm") for entry in history]
    val_lmrd = [entry.get("val_lmrd") for entry in history]
    val_mprime_loss = [entry.get("val_mprime_loss") for entry in history]
    val_of1 = [entry.get("val_of1") for entry in history]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(epochs, train_losses, marker="o", label="train_loss")
    if any(value is not None for value in val_losses):
        axes[0].plot(epochs, val_losses, marker="o", label="val_loss")
    if any(value is not None for value in train_ldfm):
        axes[0].plot(epochs, train_ldfm, linestyle="--", label="train_ldfm")
    if any(value is not None for value in train_lmrd):
        axes[0].plot(epochs, train_lmrd, linestyle="--", label="train_lmrd")
    if any(value is not None for value in train_mprime_loss):
        axes[0].plot(epochs, train_mprime_loss, linestyle="-.", label="train_mprime_loss")
    if any(value is not None for value in val_ldfm):
        axes[0].plot(epochs, val_ldfm, linestyle=":", label="val_ldfm")
    if any(value is not None for value in val_lmrd):
        axes[0].plot(epochs, val_lmrd, linestyle=":", label="val_lmrd")
    if any(value is not None for value in val_mprime_loss):
        axes[0].plot(epochs, val_mprime_loss, linestyle=(0, (3, 1, 1, 1)), label="val_mprime_loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if any(value is not None for value in val_of1):
        axes[1].plot(epochs, val_of1, marker="o", color="tab:green", label="val_oF1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation oF1")
    axes[1].grid(True, alpha=0.3)
    if any(value is not None for value in val_of1):
        axes[1].legend()

    fig.tight_layout()
    plot_path = output_dir / "metrics_plot.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def combine_datasets(dataset_list):
    if not dataset_list:
        return None
    if len(dataset_list) == 1:
        return dataset_list[0]
    return ConcatDataset(dataset_list)


def split_indices_by_label(samples, validation_split: float, seed: int):
    if validation_split <= 0.0:
        indices = list(range(len(samples)))
        return indices, []

    label_to_indices = {}
    for idx, (_, label) in enumerate(samples):
        label_to_indices.setdefault(label, []).append(idx)

    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    val_indices = []

    for indices in label_to_indices.values():
        if len(indices) < 2:
            train_indices.extend(indices)
            continue

        shuffled = [indices[i] for i in torch.randperm(len(indices), generator=generator).tolist()]
        val_count = int(round(len(shuffled) * validation_split))
        val_count = min(max(val_count, 1), len(shuffled) - 1)

        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices


def update_segmentation_counts(preds: torch.Tensor, masks: torch.Tensor, counts: dict[str, int]):
    preds_fg = preds == 1
    masks_fg = masks == 1

    counts["tp"] += int((preds_fg & masks_fg).sum().item())
    counts["fp"] += int((preds_fg & ~masks_fg).sum().item())
    counts["fn"] += int((~preds_fg & masks_fg).sum().item())
    counts["pred_pos"] += int(preds_fg.sum().item())
    counts["mask_pos"] += int(masks_fg.sum().item())
    counts["pixels"] += int(masks.numel())


def summarize_segmentation_counts(counts: dict[str, int]):
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    pixels = counts["pixels"]

    iou_den = tp + fp + fn
    dice_den = (2 * tp) + fp + fn

    return {
        "iou": (tp / iou_den) if iou_den else 0.0,
        "dice": ((2 * tp) / dice_den) if dice_den else 0.0,
        "pred_positive_rate": (counts["pred_pos"] / pixels) if pixels else 0.0,
        "mask_positive_rate": (counts["mask_pos"] / pixels) if pixels else 0.0,
    }


def split_mask_instances(mask: np.ndarray) -> list[np.ndarray]:
    if mask.ndim == 2:
        channel_masks = [(mask > 0).astype(np.uint8)]
    elif mask.ndim == 3:
        if mask.shape[0] <= 16 and mask.shape[0] <= mask.shape[-1] and mask.shape[0] <= mask.shape[-2]:
            channel_masks = [(mask[i] > 0).astype(np.uint8) for i in range(mask.shape[0])]
        elif mask.shape[-1] <= 16 and mask.shape[-1] <= mask.shape[0] and mask.shape[-1] <= mask.shape[1]:
            channel_masks = [(mask[..., i] > 0).astype(np.uint8) for i in range(mask.shape[-1])]
        else:
            raise ValueError(f"Could not infer channel axis for mask with shape {mask.shape}")
    else:
        raise ValueError(f"Unsupported mask shape {mask.shape}")

    instances = []
    for channel_mask in channel_masks:
        labeled, count = scipy.ndimage.label(channel_mask)
        for component_idx in range(1, count + 1):
            component = (labeled == component_idx).astype(np.uint8)
            if component.any():
                instances.append(component)
    return instances


def resize_binary_mask(mask: np.ndarray, image_size: int) -> np.ndarray:
    mask_tensor = torch.from_numpy(mask).view(1, 1, *mask.shape).float()
    resized = F.interpolate(mask_tensor, size=(image_size, image_size), mode="nearest")
    return resized.squeeze(0).squeeze(0).numpy().astype(np.uint8)


def load_resized_gt_instances(sample_path: str, mask_dir_by_sample: dict[str, Path | None], image_size: int) -> list[np.ndarray]:
    path = Path(sample_path)
    mask_dir = mask_dir_by_sample.get(sample_path)
    if mask_dir is None or "forged" not in path.parent.name:
        return []

    mask_path = mask_dir / path.name.replace(".png", ".npy")
    mask = np.load(mask_path)
    instances = split_mask_instances(mask)
    resized_instances = []
    for instance in instances:
        resized = resize_binary_mask(instance, image_size=image_size)
        if resized.any():
            resized_instances.append(resized)
    return resized_instances


def binary_mask_to_instances(mask: np.ndarray) -> list[np.ndarray]:
    mask = (mask > 0).astype(np.uint8)
    labeled, count = scipy.ndimage.label(mask)
    instances = []
    for component_idx in range(1, count + 1):
        component = (labeled == component_idx).astype(np.uint8)
        if component.any():
            instances.append(component)
    return instances


def calculate_binary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_flat = pred_mask.reshape(-1)
    gt_flat = gt_mask.reshape(-1)

    tp = np.sum((pred_flat == 1) & (gt_flat == 1))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return float(2 * (precision * recall) / (precision + recall))


def optimal_f1_score(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> float:
    if not pred_masks and not gt_masks:
        return 1.0
    if not pred_masks or not gt_masks:
        return 0.0

    f1_matrix = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    for pred_idx, pred_mask in enumerate(pred_masks):
        for gt_idx, gt_mask in enumerate(gt_masks):
            f1_matrix[pred_idx, gt_idx] = calculate_binary_f1(pred_mask, gt_mask)

    if f1_matrix.shape[0] < len(gt_masks):
        pad_rows = len(gt_masks) - f1_matrix.shape[0]
        f1_matrix = np.vstack((f1_matrix, np.zeros((pad_rows, f1_matrix.shape[1]), dtype=np.float32)))

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-f1_matrix)
    excess_predictions_penalty = len(gt_masks) / max(len(pred_masks), len(gt_masks))
    return float(np.mean(f1_matrix[row_ind, col_ind]) * excess_predictions_penalty)


def dice_loss(pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if pred_mask.dim() == 3:
        pred_mask = pred_mask.unsqueeze(1)
    if target_mask.dim() == 3:
        target_mask = target_mask.unsqueeze(1)

    pred_mask = pred_mask.float()
    target_mask = target_mask.float()

    pred_flat = pred_mask.reshape(pred_mask.shape[0], -1)
    target_flat = target_mask.reshape(target_mask.shape[0], -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1 - dice.mean()


def normalize_dlf_error_maps(error_maps: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if error_maps.dim() != 4:
        raise ValueError(f"Expected error maps shape [B,S,H,W], got {tuple(error_maps.shape)}")

    error_maps = torch.clamp_min(error_maps.float(), 0.0)
    error_maps = torch.log1p(error_maps)

    flattened = error_maps.flatten(start_dim=2)
    mean = flattened.mean(dim=-1, keepdim=True).unsqueeze(-1)
    std = flattened.std(dim=-1, keepdim=True, unbiased=False).unsqueeze(-1)
    normalized = (error_maps - mean) / (std + eps)
    return normalized.clamp(-6.0, 6.0)


def resize_offsets_to_image_grid(offsets: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    single_image = offsets.dim() == 3
    if single_image:
        offsets = offsets.unsqueeze(0)

    source_h, source_w = offsets.shape[-2:]
    target_h, target_w = target_size
    if (source_h, source_w) == (target_h, target_w):
        return offsets.squeeze(0) if single_image else offsets

    resized = F.interpolate(offsets, size=target_size, mode="bilinear", align_corners=True)
    scale_x = float(max(target_w - 1, 1)) / float(max(source_w - 1, 1))
    scale_y = float(max(target_h - 1, 1)) / float(max(source_h - 1, 1))
    resized[:, 0] = resized[:, 0] * scale_x
    resized[:, 1] = resized[:, 1] * scale_y
    return resized.squeeze(0) if single_image else resized


def localization_loss_terms(
    refined_mask: torch.Tensor,
    target_map: torch.Tensor,
    dlf_map: torch.Tensor,
    target_mask: torch.Tensor,
    mprime_loss_weight: float = 0.0,
):
    ldfm = dice_loss(refined_mask, target_mask)
    lmrd = dice_loss(target_map, target_mask)
    mprime_loss = dice_loss(dlf_map, target_mask)
    total_loss = ldfm + lmrd + (mprime_loss_weight * mprime_loss)
    return total_loss, ldfm, lmrd, mprime_loss


def summarize_branch_activity(dlf_map: torch.Tensor, target_map: torch.Tensor) -> dict[str, float]:
    return {
        "mprime_positive_rate": float((dlf_map >= 0.5).float().mean().item()),
        "mprime_wins_rate": float((dlf_map > target_map).float().mean().item()),
        "target_positive_rate": float((target_map >= 0.5).float().mean().item()),
    }


def extract_localization_inputs(
    images: torch.Tensor,
    pyramid_bb,
    pyramid_zm,
    feature_backbone: str,
    cnn_backbone: str,
    separate_transforms: bool,
    cnn_feature_norm: bool,
    pm_random_window: int,
    pm_iters: int,
    pm_beta: int,
    pm_use_non_local: bool,
    pm_non_local_limit: float,
    pm_reduced_precision: bool = True,
    dino_match_native_resolution: bool = False,
    collect_stats: bool = False,
):
    images_backbone = images
    if separate_transforms and feature_backbone == "cnn" and cnn_backbone == "pretrained":
        images_backbone = imagenet_normalize_tensor(images)

    device = images.device
    localization_stats = None
    peak_memory_base = None
    if collect_stats and device.type == "cuda":
        synchronize_if_cuda(device)
        torch.cuda.reset_peak_memory_stats(device)
        peak_memory_base = torch.cuda.memory_allocated(device)

    with torch.no_grad():
        if collect_stats:
            synchronize_if_cuda(device)
            feature_start = time.perf_counter()
        cnn_feats = pyramid_bb(images_backbone)
        if feature_backbone == "cnn" and cnn_backbone == "pretrained" and cnn_feature_norm:
            cnn_feats = tuple(F.normalize(feature, p=2, dim=1) for feature in cnn_feats)
        zernike_feats = pyramid_zm(images)
        if collect_stats:
            synchronize_if_cuda(device)
            feature_time = time.perf_counter() - feature_start

        patchmatch_images = images
        if feature_backbone == "dino" and dino_match_native_resolution:
            dino_match_size = cnn_feats[1].shape[-2:]
            if dino_match_size != images.shape[-2:]:
                patchmatch_images = F.interpolate(images, size=dino_match_size, mode="bilinear", align_corners=False)

        if collect_stats:
            synchronize_if_cuda(device)
            propagation_start = time.perf_counter()
        propagator = PixelPropagator(
            patchmatch_images,
            cnn_feats,
            zernike_feats,
            random_window=pm_random_window,
            reduced_precision=pm_reduced_precision,
        )
        del cnn_feats
        del zernike_feats
        batch_cnn_offsets, batch_zernike_offsets = propagator.propagation_layer(
            iters=pm_iters,
            beta=pm_beta,
            use_non_local=pm_use_non_local,
            non_local_limit=pm_non_local_limit,
        )
        if patchmatch_images.shape[-2:] != images.shape[-2:]:
            batch_cnn_offsets = resize_offsets_to_image_grid(batch_cnn_offsets, images.shape[-2:])
            batch_zernike_offsets = resize_offsets_to_image_grid(batch_zernike_offsets, images.shape[-2:])
        if collect_stats:
            synchronize_if_cuda(device)
            propagation_time = time.perf_counter() - propagation_start

        if collect_stats:
            synchronize_if_cuda(device)
            dlf_start = time.perf_counter()
        errors = MultiScaleDLF(images, batch_cnn_offsets).compute_errors()
        errors = normalize_dlf_error_maps(errors)
        if collect_stats:
            synchronize_if_cuda(device)
            dlf_time = time.perf_counter() - dlf_start
            localization_stats = {
                "feature_time_s": feature_time,
                "patchmatch_time_s": propagation_time,
                "dlf_time_s": dlf_time,
            }
            if peak_memory_base is not None:
                peak_bytes = torch.cuda.max_memory_allocated(device) - peak_memory_base
                localization_stats["localization_peak_memory_mb"] = peak_bytes / (1024 ** 2)

    return errors, batch_cnn_offsets, batch_zernike_offsets, localization_stats


def decode_and_refine_masks(
    images: torch.Tensor,
    errors: torch.Tensor,
    batch_cnn_offsets: torch.Tensor,
    batch_zernike_offsets: torch.Tensor,
    dlf_decoder,
    se_model,
):
    dlf_decoder_input = DLFDecoderInput(
        cross_scale_errors=errors,
        cnn_offsets=batch_cnn_offsets,
        zernike_offsets=batch_zernike_offsets,
    )

    dlf_map = dlf_decoder(dlf_decoder_input)
    se_input = build_se_unet_input(images, dlf_map)
    target_map = se_model(se_input)
    refined_mask = torch.maximum(dlf_map, target_map)
    return refined_mask, target_map, dlf_map


def pipeline(
    datasets=Datasets.TRAIN,
    image_size=448,
    epochs=1,
    test_run=False,
    feature_backbone="cnn",
    use_dino_transform=False,
    batch_size=4,
    dino_model_name="dinov2_vits14",
    dino_proj_dim=64,
    cnn_backbone="simple",
    cnn_pretrained_model="vgg16_bn",
    cnn_feature_norm=True,
    separate_transforms=True,
    pm_iters=16,
    pm_beta=10.0,
    pm_random_window=50,
    pm_use_non_local=False,
    pm_non_local_limit=25.0,
    pm_reduced_precision=True,
    dino_match_native_resolution=False,
    log_every=10,
    output_dir="artifacts",
    checkpoint_name="latest.pt",
    resume=True,
    save_predictions=False,
    validation_split=0.0,
    validation_seed=42,
    learning_rate=1e-3,
    mprime_loss_weight=0.0,
):
    print('[pipeline] Initializing training loop and datasets...')
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir, checkpoints_dir, predictions_dir = ensure_output_dirs(output_dir)
    checkpoint_path = checkpoints_dir / checkpoint_name
    best_checkpoint_path = checkpoints_dir / "best.pt"
    checkpoint = None
    resume_epoch = 0
    best_score = None

    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        resume_epoch = int(checkpoint.get("epoch", 0))
        best_score = checkpoint.get("best_score", checkpoint.get("best_loss"))
        print(f"[pipeline] Resuming from checkpoint: {checkpoint_path}")

    root = resolve_data_root()

    if separate_transforms:
        transform = regular_transform
    else:
        if feature_backbone == "dino" and use_dino_transform:
            transform = dino_transform
        elif feature_backbone == "cnn" and cnn_backbone == "pretrained":
            transform = imagenet_transform
        else:
            transform = regular_transform

    if batch_size > 8:
        print("[pipeline] PatchMatch is memory-heavy; forcing batch_size=8 for 16GB VRAM safety")
        batch_size = 8

    train_dataset_list = []
    val_dataset_list = []
    mask_dir_by_sample = {}
    supervised = all(dataset["masks"] is not None for dataset in datasets.value)

    for dataset_idx, dataset in enumerate(datasets.value):
        image_folder = ImageFolder(root / dataset["images"])
        samples = [(Path(path), label) for path, label in image_folder.samples]
        mask_dir = root / dataset["masks"] if dataset["masks"] is not None else None

        for sample_path, _ in samples:
            mask_dir_by_sample[str(sample_path)] = mask_dir

        train_forgery_dataset = ForgeryDataset(
            samples=samples,
            mask_dir=mask_dir,
            size=image_size,
            transform=transform,
            return_path=False,
        )

        if supervised and validation_split > 0.0:
            val_forgery_dataset = ForgeryDataset(
                samples=samples,
                mask_dir=mask_dir,
                size=image_size,
                transform=transform,
                return_path=True,
            )
            train_indices, val_indices = split_indices_by_label(
                samples,
                validation_split=validation_split,
                seed=validation_seed + dataset_idx,
            )
            train_dataset_list.append(Subset(train_forgery_dataset, train_indices))
            if val_indices:
                val_dataset_list.append(Subset(val_forgery_dataset, val_indices))
        else:
            train_dataset_list.append(train_forgery_dataset)

    train_dataset = combine_datasets(train_dataset_list)
    val_dataset = combine_datasets(val_dataset_list)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )

    print('Loading models and creating ZernikeFeatures...')
    if feature_backbone == "dino":
        from feature_extractors.dino_feature_extractor import PyramidDinoFeatureExtractor

        pyramid_bb = PyramidDinoFeatureExtractor(
            model_name=dino_model_name,
            normalize_input=True if separate_transforms else not use_dino_transform,
            proj_dim=dino_proj_dim,
            upsample_to_input=not dino_match_native_resolution,
        ).to(device)
    else:
        if cnn_backbone == "pretrained":
            backbone = PretrainedBackboneExtractor(model_name=cnn_pretrained_model, out_dim=32, freeze=True)
            pyramid_bb = PyramidFeatureExtractor(backbone=backbone).to(device)
        else:
            pyramid_bb = PyramidFeatureExtractor().to(device)

    pq_list = default_pq_list(max_order=5)
    pyramid_zm = PyramidZernikeExtractor(pq_list, kernel_size=13).to(device)
    pyramid_bb.eval()
    pyramid_zm.eval()

    dlf_decoder = None
    se_model = SEUNet(in_channels=4, out_channels=1, final_activation="sigmoid").to(device)
    optimizer = None

    if test_run or not supervised:
        se_model.eval()
    else:
        se_model.train()

    batch_counter = 0
    run_start = time.perf_counter()
    eta_batch_times = []
    eta_printed = False

    print('Starting training...')

    for epoch_idx in range(resume_epoch, epochs):
        epoch_start = time.perf_counter()
        epoch_loss_sum = 0.0
        epoch_ldfm_sum = 0.0
        epoch_lmrd_sum = 0.0
        epoch_mprime_loss_sum = 0.0
        epoch_mprime_positive_rate_sum = 0.0
        epoch_mprime_wins_rate_sum = 0.0
        epoch_loss_steps = 0

        if optimizer is not None:
            dlf_decoder.train()
            se_model.train()

        for batch_idx, (images, masks, labels) in enumerate(train_loader, start=1):
            batch_start = time.perf_counter()
            images = images.to(device)
            masks = masks.to(device=device, dtype=torch.float32)
            labels = labels.to(device)

            if supervised:
                mask_min = int(masks.min().item())
                mask_max = int(masks.max().item())
                if mask_min < 0 or mask_max > 1:
                    raise ValueError(
                        f"Dice loss expects binary target values in [0, 1], got range [{mask_min}, {mask_max}] "
                        f"at epoch {epoch_idx + 1}, batch {batch_idx}."
                    )

            collect_localization_stats = log_every > 0 and batch_idx % log_every == 0
            errors, batch_cnn_offsets, batch_zernike_offsets, localization_stats = extract_localization_inputs(
                images=images,
                pyramid_bb=pyramid_bb,
                pyramid_zm=pyramid_zm,
                feature_backbone=feature_backbone,
                cnn_backbone=cnn_backbone,
                separate_transforms=separate_transforms,
                cnn_feature_norm=cnn_feature_norm,
                pm_random_window=pm_random_window,
                pm_iters=pm_iters,
                pm_beta=pm_beta,
                pm_use_non_local=pm_use_non_local,
                pm_non_local_limit=pm_non_local_limit,
                pm_reduced_precision=pm_reduced_precision,
                dino_match_native_resolution=dino_match_native_resolution,
                collect_stats=collect_localization_stats,
            )

            if dlf_decoder is None:
                dlf_decoder = DLFDecoder(num_error_maps=errors.shape[1]).to(device)
                if test_run or not supervised:
                    dlf_decoder.eval()
                else:
                    dlf_decoder.train()
                    optimizer = torch.optim.Adam(
                        list(dlf_decoder.parameters()) + list(se_model.parameters()),
                        lr=learning_rate,
                    )

                if checkpoint is not None:
                    dlf_decoder.load_state_dict(checkpoint["dlf_decoder"])
                    se_model.load_state_dict(checkpoint["se_model"])
                    if optimizer is not None and checkpoint.get("optimizer") is not None:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                        set_optimizer_learning_rate(optimizer, learning_rate)
                    checkpoint = None

            refined_mask, target_map, dlf_map = decode_and_refine_masks(
                images=images,
                errors=errors,
                batch_cnn_offsets=batch_cnn_offsets,
                batch_zernike_offsets=batch_zernike_offsets,
                dlf_decoder=dlf_decoder,
                se_model=se_model,
            )

            if test_run:
                display_image(images[0], masks[0])
                display_pixel_offsets(batch_cnn_offsets[0], batch_zernike_offsets[0], images[0])
                display_image(images[0], (refined_mask[0, 0] >= 0.5).long())
                return

            if optimizer is not None:
                optimizer.zero_grad()
                loss, ldfm, lmrd, mprime_loss = localization_loss_terms(
                    refined_mask,
                    target_map,
                    dlf_map,
                    masks,
                    mprime_loss_weight=mprime_loss_weight,
                )
                branch_stats = summarize_branch_activity(dlf_map, target_map)
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                ldfm_value = ldfm.item()
                lmrd_value = lmrd.item()
                mprime_loss_value = mprime_loss.item()
                epoch_loss_sum += loss_value
                epoch_ldfm_sum += ldfm_value
                epoch_lmrd_sum += lmrd_value
                epoch_mprime_loss_sum += mprime_loss_value
                epoch_mprime_positive_rate_sum += branch_stats["mprime_positive_rate"]
                epoch_mprime_wins_rate_sum += branch_stats["mprime_wins_rate"]
                epoch_loss_steps += 1

                if log_every > 0 and batch_idx % log_every == 0:
                    localization_message = ""
                    if localization_stats is not None:
                        localization_message = (
                            f" feat: {localization_stats['feature_time_s']:.2f}s"
                            f" pm: {localization_stats['patchmatch_time_s']:.2f}s"
                            f" dlf: {localization_stats['dlf_time_s']:.2f}s"
                        )
                        peak_memory_mb = localization_stats.get("localization_peak_memory_mb")
                        if peak_memory_mb is not None:
                            localization_message += f" loc_peak: {peak_memory_mb:.0f}MB"
                    print(
                        f"[epoch {epoch_idx + 1}/{epochs}] "
                        f"batch {batch_idx}/{len(train_loader)} "
                        f"loss: {loss_value:.4f} "
                        f"(ldfm={ldfm_value:.4f}, lmrd={lmrd_value:.4f}, mprime={mprime_loss_value:.4f}, lambda={mprime_loss_weight:.2f}) "
                        f"mprime_pos: {branch_stats['mprime_positive_rate']:.4%} "
                        f"mprime_wins: {branch_stats['mprime_wins_rate']:.4%} "
                        f"time spent: {(time.perf_counter() - batch_start):.2f}"
                        f"{localization_message}"
                    )

                if epoch_idx == resume_epoch and batch_idx > 5:
                    eta_batch_times.append(time.perf_counter() - batch_start)
                    if not eta_printed and len(eta_batch_times) >= 5:
                        avg_batch_s = sum(eta_batch_times) / len(eta_batch_times)
                        est_train_total_s = avg_batch_s * epochs * len(train_loader)
                        print(
                            f"[pipeline] Warmed-up train-only estimate: "
                            f"{est_train_total_s / 60:.2f} minutes or {est_train_total_s / 3600:.2f} hours"
                        )
                        eta_printed = True

            mask_preds = (refined_mask >= 0.5).long().squeeze(1)
            if save_predictions:
                save_prediction_batch(predictions_dir, epoch_idx, batch_counter + 1, mask_preds)

            batch_counter += 1

        val_metrics = None
        if val_loader is not None and dlf_decoder is not None:
            dlf_decoder.eval()
            se_model.eval()
            val_loss_sum = 0.0
            val_ldfm_sum = 0.0
            val_lmrd_sum = 0.0
            val_mprime_loss_sum = 0.0
            val_mprime_positive_rate_sum = 0.0
            val_mprime_wins_rate_sum = 0.0
            val_loss_steps = 0
            val_counts = {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "pred_pos": 0,
                "mask_pos": 0,
                "pixels": 0,
            }
            val_of1_sum = 0.0
            val_images = 0

            with torch.no_grad():
                for images, masks, labels, image_paths in val_loader:
                    images = images.to(device)
                    masks = masks.to(device=device, dtype=torch.float32)
                    labels = labels.to(device)

                    errors, batch_cnn_offsets, batch_zernike_offsets, _ = extract_localization_inputs(
                        images=images,
                        pyramid_bb=pyramid_bb,
                        pyramid_zm=pyramid_zm,
                        feature_backbone=feature_backbone,
                        cnn_backbone=cnn_backbone,
                        separate_transforms=separate_transforms,
                        cnn_feature_norm=cnn_feature_norm,
                        pm_random_window=pm_random_window,
                        pm_iters=pm_iters,
                        pm_beta=pm_beta,
                        pm_use_non_local=pm_use_non_local,
                        pm_non_local_limit=pm_non_local_limit,
                        pm_reduced_precision=pm_reduced_precision,
                        dino_match_native_resolution=dino_match_native_resolution,
                    )
                    refined_mask, target_map, dlf_map = decode_and_refine_masks(
                        images=images,
                        errors=errors,
                        batch_cnn_offsets=batch_cnn_offsets,
                        batch_zernike_offsets=batch_zernike_offsets,
                        dlf_decoder=dlf_decoder,
                        se_model=se_model,
                    )

                    val_loss, val_ldfm, val_lmrd, val_mprime_loss = localization_loss_terms(
                        refined_mask,
                        target_map,
                        dlf_map,
                        masks,
                        mprime_loss_weight=mprime_loss_weight,
                    )
                    branch_stats = summarize_branch_activity(dlf_map, target_map)
                    val_loss_sum += val_loss.item()
                    val_ldfm_sum += val_ldfm.item()
                    val_lmrd_sum += val_lmrd.item()
                    val_mprime_loss_sum += val_mprime_loss.item()
                    val_mprime_positive_rate_sum += branch_stats["mprime_positive_rate"]
                    val_mprime_wins_rate_sum += branch_stats["mprime_wins_rate"]
                    val_loss_steps += 1

                    mask_preds = (refined_mask >= 0.5).long().squeeze(1)
                    update_segmentation_counts(mask_preds, masks.long(), val_counts)

                    pred_masks_np = mask_preds.cpu().numpy().astype(np.uint8)
                    for pred_mask, image_path in zip(pred_masks_np, image_paths):
                        pred_instances = binary_mask_to_instances(pred_mask)
                        gt_instances = load_resized_gt_instances(
                            image_path,
                            mask_dir_by_sample=mask_dir_by_sample,
                            image_size=image_size,
                        )
                        val_of1_sum += optimal_f1_score(pred_instances, gt_instances)
                        val_images += 1

            val_metrics = summarize_segmentation_counts(val_counts)
            val_mean_loss = val_loss_sum / max(val_loss_steps, 1)
            val_mean_ldfm = val_ldfm_sum / max(val_loss_steps, 1)
            val_mean_lmrd = val_lmrd_sum / max(val_loss_steps, 1)
            val_mean_mprime_loss = val_mprime_loss_sum / max(val_loss_steps, 1)
            val_mean_mprime_positive_rate = val_mprime_positive_rate_sum / max(val_loss_steps, 1)
            val_mean_mprime_wins_rate = val_mprime_wins_rate_sum / max(val_loss_steps, 1)
            val_mean_of1 = val_of1_sum / max(val_images, 1)
            print(
                f"[epoch {epoch_idx + 1}/{epochs}] "
                f"val_loss: {val_mean_loss:.4f} "
                f"(ldfm={val_mean_ldfm:.4f}, lmrd={val_mean_lmrd:.4f}, mprime={val_mean_mprime_loss:.4f}, lambda={mprime_loss_weight:.2f}) "
                f"val_oF1: {val_mean_of1:.4f} "
                f"val_pred_pos: {val_metrics['pred_positive_rate']:.4%} "
                f"val_mprime_pos: {val_mean_mprime_positive_rate:.4%} "
                f"val_mprime_wins: {val_mean_mprime_wins_rate:.4%}"
            )

            if optimizer is not None:
                dlf_decoder.train()
                se_model.train()

        if epoch_loss_steps > 0:
            mean_loss = epoch_loss_sum / epoch_loss_steps
            mean_ldfm = epoch_ldfm_sum / epoch_loss_steps
            mean_lmrd = epoch_lmrd_sum / epoch_loss_steps
            mean_mprime_loss = epoch_mprime_loss_sum / epoch_loss_steps
            mean_mprime_positive_rate = epoch_mprime_positive_rate_sum / epoch_loss_steps
            mean_mprime_wins_rate = epoch_mprime_wins_rate_sum / epoch_loss_steps
            metrics = {
                "epoch": epoch_idx + 1,
                "train_loss": mean_loss,
                "train_ldfm": mean_ldfm,
                "train_lmrd": mean_lmrd,
                "train_mprime_loss": mean_mprime_loss,
                "train_mprime_positive_rate": mean_mprime_positive_rate,
                "train_mprime_wins_rate": mean_mprime_wins_rate,
                "steps": epoch_loss_steps,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }

            if val_metrics is not None:
                metrics["val_loss"] = val_mean_loss
                metrics["val_ldfm"] = val_mean_ldfm
                metrics["val_lmrd"] = val_mean_lmrd
                metrics["val_mprime_loss"] = val_mean_mprime_loss
                metrics["val_mprime_positive_rate"] = val_mean_mprime_positive_rate
                metrics["val_mprime_wins_rate"] = val_mean_mprime_wins_rate
                metrics["val_of1"] = val_mean_of1
                metrics["val_iou"] = val_metrics["iou"]
                metrics["val_dice"] = val_metrics["dice"]
                metrics["val_pred_positive_rate"] = val_metrics["pred_positive_rate"]
                metrics["val_mask_positive_rate"] = val_metrics["mask_positive_rate"]

            print(
                f"[epoch {epoch_idx + 1}/{epochs}] "
                f"train_loss: {mean_loss:.4f} "
                f"(ldfm={mean_ldfm:.4f}, lmrd={mean_lmrd:.4f}, mprime={mean_mprime_loss:.4f}, lambda={mprime_loss_weight:.2f}) "
                f"mprime_pos: {mean_mprime_positive_rate:.4%} "
                f"mprime_wins: {mean_mprime_wins_rate:.4%} "
                f"completed in: {metrics['epoch_seconds']:.2f}s"
            )
            append_metrics_log(output_dir, metrics)
            save_metrics_plot(output_dir)

            if optimizer is not None:
                checkpoint_score = val_mean_of1 if val_metrics is not None else -mean_loss
                save_checkpoint(
                    checkpoint_path,
                    epoch=epoch_idx + 1,
                    dlf_decoder=dlf_decoder,
                    se_model=se_model,
                    optimizer=optimizer,
                    best_score=checkpoint_score,
                )
                if best_score is None or checkpoint_score > best_score:
                    best_score = checkpoint_score
                    save_checkpoint(
                        best_checkpoint_path,
                        epoch=epoch_idx + 1,
                        dlf_decoder=dlf_decoder,
                        se_model=se_model,
                        optimizer=optimizer,
                        best_score=best_score,
                    )

    save_metrics_plot(output_dir)
    print(f"Training completed. Total time: {time.perf_counter() - run_start:.2f}s")
