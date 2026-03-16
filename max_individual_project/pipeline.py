from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dataset import ForgeryDataset, Datasets, regular_transform, dino_transform, imagenet_transform
import time

from feature_extractors.cnn_feature_extractor import PyramidFeatureExtractor, PretrainedBackboneExtractor
from feature_extractors.zernike_feature_extractor import PyramidZernikeExtractor, default_pq_list
from cross_scale_patternmatch.pixel_propagator import PixelPropagator

from visualizer import *

def pipeline(
    datasets=Datasets.TRAIN,
    image_size=488,
    test_run=False,
    feature_backbone="cnn",
    use_dino_transform=False,
    batch_size=8,
    dino_model_name="dinov2_vits14",
    dino_proj_dim=64,
    cnn_backbone="simple",
    cnn_pretrained_model="vgg16_bn",
    cnn_feature_norm=True,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    root = Path('data')

    # Choose transform based on backbone
    if feature_backbone == "dino" and use_dino_transform:
        transform = dino_transform
    elif feature_backbone == "cnn" and cnn_backbone == "pretrained":
        transform = imagenet_transform
    else:
        transform = regular_transform

    if feature_backbone == "dino" and batch_size > 4:
        print("[pipeline] DINO backbone is memory heavy; forcing batch_size=4")
        batch_size = 4

    for dataset in datasets.value:
        image_folder = ImageFolder(root / dataset['images'])

        samples = [(Path(p), y) for p, y in image_folder.samples]

        forgery_dataset = ForgeryDataset(
            samples=samples,
            mask_dir=root / dataset['masks'] if dataset['masks'] is not None else None,
            size=image_size,
            transform=transform
        )

    train_loader = DataLoader(forgery_dataset, batch_size=batch_size, shuffle=True)

    # Feature extraction
    if feature_backbone == "dino":
        from feature_extractors.dino_feature_extractor import PyramidDinoFeatureExtractor
        pyramid_bb = PyramidDinoFeatureExtractor(
            model_name=dino_model_name,
            normalize_input=not use_dino_transform,
            proj_dim=dino_proj_dim,
        ).to(device)
    else:
        if cnn_backbone == "pretrained":
            backbone = PretrainedBackboneExtractor(model_name=cnn_pretrained_model, out_dim=32, freeze=True)
            pyramid_bb = PyramidFeatureExtractor(backbone=backbone).to(device)
        else:
            pyramid_bb = PyramidFeatureExtractor().to(device)

    # Zernike pair things
    pq_list = default_pq_list(max_order=5)
    pyramid_zm = PyramidZernikeExtractor(pq_list, kernel_size=13).to(device)

    # MAIN LOOP
    batch_counter = 0

    for images, masks, labels in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            cnn_feats = pyramid_bb(images)
            if feature_backbone == "cnn" and cnn_backbone == "pretrained" and cnn_feature_norm:
                cnn_feats = tuple(F.normalize(f, p=2, dim=1) for f in cnn_feats)
            zernike_feats = pyramid_zm(images)

        for idx, img in enumerate(images):
            img_cnn_feats = tuple(f[idx] for f in cnn_feats)
            img_zernike_feats = tuple(f[idx] for f in zernike_feats)
            propagator = PixelPropagator(img, img_cnn_feats, img_zernike_feats)
            res = propagator.propagation_layer(iters=32)

            if test_run: 
                display_image(img, masks[idx])
                display_pixel_offsets(res[0], res[1], img)
                return

        del cnn_feats, zernike_feats
        batch_counter += 1

        print(f'Processed batch: {batch_counter} at: {time.perf_counter():.2f}')
