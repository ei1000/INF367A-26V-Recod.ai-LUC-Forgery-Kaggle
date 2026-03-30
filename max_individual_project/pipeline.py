from pathlib import Path
import json
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset, DataLoader
from dataset import ForgeryDataset, Datasets, regular_transform, dino_transform, imagenet_transform
import time

from datatypes import DLFDecoderInput

from feature_extractors.cnn_feature_extractor import PyramidFeatureExtractor, PretrainedBackboneExtractor
from feature_extractors.zernike_feature_extractor import PyramidZernikeExtractor, default_pq_list
from cross_scale_patternmatch.pixel_propagator import PixelPropagator

from prediction.decoder import DLFDecoder
from prediction.multi_scale_dlf import MultiScaleDLF
from prediction.se_u_net import build_se_unet_input, SEUNet

try:
    from visualizer import display_image, display_pixel_offsets
except ModuleNotFoundError:
    def display_image(*args, **kwargs):
        raise RuntimeError("Visualization requires optional dependency 'matplotlib'.")

    def display_pixel_offsets(*args, **kwargs):
        raise RuntimeError("Visualization requires optional dependency 'matplotlib'.")


def _resolve_data_root() -> Path:
    root = Path("data")
    if root.exists():
        return root
    alt = Path(__file__).resolve().parent.parent / "data"
    if alt.exists():
        return alt
    raise FileNotFoundError("Could not find data directory. Checked ./data and ../data.")


def _imagenet_normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _ensure_output_dirs(output_dir: str | Path):
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    predictions_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, checkpoints_dir, predictions_dir


def _save_checkpoint(path: Path, epoch: int, dlf_decoder, se_model, optimizer, best_loss: float | None):
    checkpoint = {
        "epoch": epoch,
        "dlf_decoder": dlf_decoder.state_dict(),
        "se_model": se_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "best_loss": best_loss,
    }
    torch.save(checkpoint, path)


def _save_prediction_batch(predictions_dir: Path, epoch_idx: int, batch_idx: int, mask_preds: torch.Tensor):
    pred_path = predictions_dir / f"epoch_{epoch_idx + 1:03d}_batch_{batch_idx:05d}.pt"
    torch.save(mask_preds.detach().cpu(), pred_path)


def _append_metrics_log(output_dir: Path, metrics: dict):
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")


def pipeline(
    datasets=Datasets.TRAIN,
    image_size=448,
    epochs=1,
    test_run=False,
    feature_backbone="cnn",
    use_dino_transform=False,
    batch_size=1,
    dino_model_name="dinov2_vits14",
    dino_proj_dim=64,
    cnn_backbone="simple",
    cnn_pretrained_model="vgg16_bn",
    cnn_feature_norm=True,
    separate_transforms=True,
    pm_iters=16,
    pm_beta=1000,
    pm_random_window=50,
    pm_use_non_local=False,
    pm_non_local_limit=25.0,
    log_every=10,
    output_dir="artifacts",
    checkpoint_name="latest.pt",
    resume=True,
    save_predictions=True,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir, checkpoints_dir, predictions_dir = _ensure_output_dirs(output_dir)
    checkpoint_path = checkpoints_dir / checkpoint_name
    best_checkpoint_path = checkpoints_dir / "best.pt"
    checkpoint = None
    resume_epoch = 0
    best_loss = None

    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        resume_epoch = int(checkpoint.get("epoch", 0))
        best_loss = checkpoint.get("best_loss")
        print(f"[pipeline] Resuming from checkpoint: {checkpoint_path}")

    # Load data
    root = _resolve_data_root()

    # Choose dataset transform (raw if we want separate transforms)
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

    dataset_list = []
    for dataset in datasets.value:
        image_folder = ImageFolder(root / dataset['images'])

        samples = [(Path(p), y) for p, y in image_folder.samples]

        dataset_list.append(ForgeryDataset(
            samples=samples,
            mask_dir=root / dataset['masks'] if dataset['masks'] is not None else None,
            size=image_size,
            transform=transform
        ))

    if len(dataset_list) == 1:
        forgery_dataset = dataset_list[0]
    else:
        forgery_dataset = ConcatDataset(dataset_list)

    train_loader = DataLoader(forgery_dataset, batch_size=batch_size, shuffle=True)

    # Feature extraction
    if feature_backbone == "dino":
        from feature_extractors.dino_feature_extractor import PyramidDinoFeatureExtractor
        pyramid_bb = PyramidDinoFeatureExtractor(
            model_name=dino_model_name,
            normalize_input=True if separate_transforms else not use_dino_transform,
            proj_dim=dino_proj_dim,
        ).to(device)
    else:
        if cnn_backbone == "pretrained":
            backbone = PretrainedBackboneExtractor(model_name=cnn_pretrained_model, out_dim=32, freeze=True)
            pyramid_bb = PyramidFeatureExtractor(backbone=backbone).to(device)
        else:
            pyramid_bb = PyramidFeatureExtractor().to(device)

    # Zernike pairs
    pq_list = default_pq_list(max_order=5)
    pyramid_zm = PyramidZernikeExtractor(pq_list, kernel_size=13).to(device)
    pyramid_bb.eval()
    pyramid_zm.eval()
    dlf_decoder = None
    se_model = SEUNet(in_channels=4, out_channels=2).to(device)
    supervised = all(dataset["masks"] is not None for dataset in datasets.value)
    loss_fn = torch.nn.CrossEntropyLoss() if supervised else None
    optimizer = None

    if test_run or not supervised:
        se_model.eval()
    else:
        se_model.train()

    # MAIN LOOP
    batch_counter = 0

    start_time = time.perf_counter()

    for epoch_idx in range(resume_epoch, epochs):
        epoch_loss_sum = 0.0
        epoch_loss_steps = 0

        if optimizer is not None:
            dlf_decoder.train()
            se_model.train()

        for batch_idx, (images, masks, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            masks = masks.to(device=device, dtype=torch.long)
            labels = labels.to(device)
            if supervised:
                mask_min = int(masks.min().item())
                mask_max = int(masks.max().item())
                if mask_min < 0 or mask_max > 1:
                    raise ValueError(
                        f"CrossEntropyLoss expects target classes in [0, 1], got range [{mask_min}, {mask_max}] "
                        f"at epoch {epoch_idx + 1}, batch {batch_idx}."
                    )

            # Separate transforms: CNN/DINO can be normalized while Zernike stays raw
            images_backbone = images
            if separate_transforms and feature_backbone == "cnn" and cnn_backbone == "pretrained":
                images_backbone = _imagenet_normalize_tensor(images)

            with torch.no_grad():
                cnn_feats = pyramid_bb(images_backbone)
                if feature_backbone == "cnn" and cnn_backbone == "pretrained" and cnn_feature_norm:
                    cnn_feats = tuple(F.normalize(f, p=2, dim=1) for f in cnn_feats)
                zernike_feats = pyramid_zm(images)

            batch_cnn_offsets = []
            batch_zernike_offsets = []
            for idx, img in enumerate(images):
                img_cnn_feats = tuple(f[idx] for f in cnn_feats)
                img_zernike_feats = tuple(f[idx] for f in zernike_feats)
                propagator = PixelPropagator(img, img_cnn_feats, img_zernike_feats, random_window=pm_random_window)
                cnn_offsets, zernike_offsets = propagator.propagation_layer(
                    iters=pm_iters,
                    beta=pm_beta,
                    use_non_local=pm_use_non_local,
                    non_local_limit=pm_non_local_limit,
                )

                if test_run:
                    display_image(img, masks[idx])
                    display_pixel_offsets(cnn_offsets, zernike_offsets, img)

                batch_cnn_offsets.append(cnn_offsets)
                batch_zernike_offsets.append(zernike_offsets)

            batch_cnn_offsets = torch.stack(batch_cnn_offsets, dim=0)
            batch_zernike_offsets = torch.stack(batch_zernike_offsets, dim=0)

            dense_linear_fitter = MultiScaleDLF(images, batch_cnn_offsets)
            errors = dense_linear_fitter.compute_errors()

            if dlf_decoder is None:
                dlf_decoder = DLFDecoder(num_error_maps=errors.shape[1]).to(device)
                if test_run or not supervised:
                    dlf_decoder.eval()
                else:
                    dlf_decoder.train()
                    optimizer = torch.optim.Adam(
                        list(dlf_decoder.parameters()) + list(se_model.parameters()),
                        lr=1e-3,
                    )
                if checkpoint is not None:
                    dlf_decoder.load_state_dict(checkpoint["dlf_decoder"])
                    se_model.load_state_dict(checkpoint["se_model"])
                    if optimizer is not None and checkpoint.get("optimizer") is not None:
                        optimizer.load_state_dict(checkpoint["optimizer"])
                    checkpoint = None

            dlf_decoder_input = DLFDecoderInput(
                cross_scale_errors=errors,
                cnn_offsets=batch_cnn_offsets,
                zernike_offsets=batch_zernike_offsets,
            )

            if optimizer is not None:
                optimizer.zero_grad()
                dlf_map = dlf_decoder(dlf_decoder_input)
                se_input = build_se_unet_input(images, dlf_map)
                mask_logits = se_model(se_input)
                loss = loss_fn(mask_logits, masks)
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                epoch_loss_sum += loss_value
                epoch_loss_steps += 1
                if log_every > 0 and batch_idx % log_every == 0:
                    print(
                        f"[epoch {epoch_idx + 1}/{epochs}] "
                        f"batch {batch_idx}/{len(train_loader)} "
                        f"loss: {loss_value:.4f}"
                        f"time spent: {(time.perf_counter() - start_time):.2f}"
                    )
            else:
                with torch.no_grad():
                    dlf_map = dlf_decoder(dlf_decoder_input)
                    se_input = build_se_unet_input(images, dlf_map)
                    mask_logits = se_model(se_input)

            mask_preds = mask_logits.argmax(dim=1)
            if save_predictions:
                _save_prediction_batch(predictions_dir, epoch_idx, batch_counter + 1, mask_preds)

            if test_run:
                display_image(images[0], mask_preds[0])
                return

            del cnn_feats, zernike_feats
            batch_counter += 1


        if epoch_loss_steps > 0:
            mean_loss = epoch_loss_sum / epoch_loss_steps
            print(f"[epoch {epoch_idx + 1}/{epochs}] mean loss: {mean_loss:.4f} completed in: {(time.perf_counter() - start_time):.2f}")
            _append_metrics_log(output_dir, {
                "epoch": epoch_idx + 1,
                "mean_loss": mean_loss,
                "steps": epoch_loss_steps,
            })
            if optimizer is not None:
                _save_checkpoint(
                    checkpoint_path,
                    epoch=epoch_idx + 1,
                    dlf_decoder=dlf_decoder,
                    se_model=se_model,
                    optimizer=optimizer,
                    best_loss=best_loss,
                )
                if best_loss is None or mean_loss < best_loss:
                    best_loss = mean_loss
                    _save_checkpoint(
                        best_checkpoint_path,
                        epoch=epoch_idx + 1,
                        dlf_decoder=dlf_decoder,
                        se_model=se_model,
                        optimizer=optimizer,
                        best_loss=best_loss,
                    )
    print(f'Training completed. Total time: {time.perf_counter() - start_time}')
