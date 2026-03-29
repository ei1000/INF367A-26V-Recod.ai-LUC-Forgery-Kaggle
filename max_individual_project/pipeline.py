from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset, DataLoader
from dataset import ForgeryDataset, Datasets, regular_transform, dino_transform, imagenet_transform
import time

from feature_extractors.cnn_feature_extractor import PyramidFeatureExtractor, PretrainedBackboneExtractor
from feature_extractors.zernike_feature_extractor import PyramidZernikeExtractor, default_pq_list
from cross_scale_patternmatch.pixel_propagator import PixelPropagator

from max_individual_project.prediction.decoder import DLFDecoder
from prediction.multi_scale_dlf import MultiScaleDLF

from visualizer import *


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


def pipeline(
    datasets=Datasets.TRAIN,
    image_size=488,
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
    pm_iters=32,
    pm_beta=1000,
    pm_random_window=50,
    pm_use_non_local=False,
    pm_non_local_limit=25.0,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    train_loader = DataLoader(forgery_dataset, batch_size=batch_size, shuffle=False)

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

    # MAIN LOOP
    batch_counter = 0

    for images, masks, labels in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        # Separate transforms: CNN/DINO can be normalized while Zernike stays raw
        images_backbone = images
        if separate_transforms and feature_backbone == "cnn" and cnn_backbone == "pretrained":
            images_backbone = _imagenet_normalize_tensor(images)

        with torch.no_grad():
            cnn_feats = pyramid_bb(images_backbone)
            if feature_backbone == "cnn" and cnn_backbone == "pretrained" and cnn_feature_norm:
                cnn_feats = tuple(F.normalize(f, p=2, dim=1) for f in cnn_feats)
            zernike_feats = pyramid_zm(images)

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
                return

            # TODO: DLF + build for batching support
            dense_linear_fitter = MultiScaleDLF(img, cnn_offsets)
            errors = dense_linear_fitter.compute_errors()

            # TODO: Decoder
            dlf_decoder = DLFDecoder(errors, cnn_offsets, zernike_offsets)
            # dlf_decoder.predict()

            # TODO: SE-U-Net + argmax preds


        del cnn_feats, zernike_feats
        batch_counter += 1

        print(f'Processed batch: {batch_counter} at: {time.perf_counter():.2f}')
