from pathlib import Path
import torch
import torch.nn.functional as F

from dataset import ForgeryDataset, regular_transform, dino_transform, imagenet_transform
from feature_extractors.cnn_feature_extractor import PyramidFeatureExtractor, PretrainedBackboneExtractor
from feature_extractors.zernike_feature_extractor import PyramidZernikeExtractor, default_pq_list
from cross_scale_patternmatch.pixel_propagator import PixelPropagator
from visualizer import display_image, display_pixel_offsets


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


def _build_transform(feature_backbone: str, use_dino_transform: bool, cnn_backbone: str, separate_transforms: bool):
    if separate_transforms:
        return regular_transform
    if feature_backbone == "dino" and use_dino_transform:
        return dino_transform
    if feature_backbone == "cnn" and cnn_backbone == "pretrained":
        return imagenet_transform
    return regular_transform


def _build_backbone(
    feature_backbone: str,
    cnn_backbone: str,
    cnn_pretrained_model: str,
    dino_model_name: str,
    dino_proj_dim: int | None,
    use_dino_transform: bool,
    separate_transforms: bool,
    device: str,
):
    if feature_backbone == "dino":
        from feature_extractors.dino_feature_extractor import PyramidDinoFeatureExtractor
        return PyramidDinoFeatureExtractor(
            model_name=dino_model_name,
            normalize_input=True if separate_transforms else not use_dino_transform,
            proj_dim=dino_proj_dim,
        ).to(device)

    if cnn_backbone == "pretrained":
        backbone = PretrainedBackboneExtractor(model_name=cnn_pretrained_model, out_dim=32, freeze=True)
        return PyramidFeatureExtractor(backbone=backbone).to(device)

    return PyramidFeatureExtractor().to(device)


def run_compare(
    image_size=488,
    feature_backbone="cnn",
    cnn_backbone="pretrained",
    cnn_pretrained_model="vgg16_bn",
    dino_model_name="dinov2_vits14",
    dino_proj_dim=64,
    use_dino_transform=False,
    cnn_feature_norm=True,
    iters=24,
    beta=2.5,
    pm_random_window=50,
    pm_use_non_local=False,
    pm_non_local_limit=25.0,
    separate_transforms=True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = _resolve_data_root()
    authentic_path = root / "train_images" / "authentic" / "10.png"
    forged_path = root / "train_images" / "forged" / "10.png"

    transform = _build_transform(feature_backbone, use_dino_transform, cnn_backbone, separate_transforms)

    samples = [(authentic_path, 0), (forged_path, 1)]
    dataset = ForgeryDataset(
        samples=samples,
        mask_dir=root / "train_masks",
        size=image_size,
        transform=transform,
    )

    pyramid_bb = _build_backbone(
        feature_backbone=feature_backbone,
        cnn_backbone=cnn_backbone,
        cnn_pretrained_model=cnn_pretrained_model,
        dino_model_name=dino_model_name,
        dino_proj_dim=dino_proj_dim,
        use_dino_transform=use_dino_transform,
        separate_transforms=separate_transforms,
        device=device,
    )

    pq_list = default_pq_list(max_order=5)
    pyramid_zm = PyramidZernikeExtractor(pq_list, kernel_size=13).to(device)

    labels = ["authentic/10.png", "forged/10.png"]

    for idx in range(len(dataset)):
        img, mask, _ = dataset[idx]
        images = img.unsqueeze(0).to(device)

        images_backbone = images
        if separate_transforms and feature_backbone == "cnn" and cnn_backbone == "pretrained":
            images_backbone = _imagenet_normalize_tensor(images)

        with torch.no_grad():
            cnn_feats = pyramid_bb(images_backbone)
            if feature_backbone == "cnn" and cnn_backbone == "pretrained" and cnn_feature_norm:
                cnn_feats = tuple(F.normalize(f, p=2, dim=1) for f in cnn_feats)
            zernike_feats = pyramid_zm(images)

        img_cnn_feats = tuple(f[0] for f in cnn_feats)
        img_zernike_feats = tuple(f[0] for f in zernike_feats)

        propagator = PixelPropagator(images[0], img_cnn_feats, img_zernike_feats, random_window=pm_random_window)
        cnn_offsets, zernike_offsets = propagator.propagation_layer(
            iters=iters,
            beta=beta,
            use_non_local=pm_use_non_local,
            non_local_limit=pm_non_local_limit,
        )

        print(f"[compare] {labels[idx]} - iters={iters}, beta={beta}, non_local={pm_use_non_local}")
        display_image(images[0], mask)
        display_pixel_offsets(cnn_offsets, zernike_offsets, images[0])


if __name__ == "__main__":
    run_compare()
