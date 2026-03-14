from pathlib import Path
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dataset import ForgeryDataset
import time

from feature_extractors.cnn_feature_extractor import PyramidFeatureExtractor
from feature_extractors.zernike_feature_extractor import PyramidZernikeExtractor
from cross_scale_patternmatch.pixel_propagator import PixelPropagator


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
root = Path('data')
train_image_folder = ImageFolder(root / "train_images")

samples = [(Path(p), y) for p, y in train_image_folder.samples]

train_dataset = ForgeryDataset(
    samples=samples,
    mask_dir=root / "train_masks",
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)

# Feature extraction
pyramid_bb = PyramidFeatureExtractor().to(device)

# Zernike pair things
pq_list = [(0,0),(1,1),(0,2),(1,3),(0,4),(1,5)]
pyramid_zm = PyramidZernikeExtractor(pq_list, kernel_size=64).to(device)

batch_counter = 0


for images, _, _ in train_loader:
    images = images.to(device)
    with torch.no_grad():
        cnn_feats = pyramid_bb(images)
        zernike_feats = pyramid_zm(images)

    for idx, img in enumerate(images):
        img_cnn_feats = tuple(f[idx] for f in cnn_feats)
        img_zernike_feats = tuple(f[idx] for f in zernike_feats)
        propagator = PixelPropagator(img, img_cnn_feats, img_zernike_feats)
        res = propagator.propagation_layer()
        print(res)

    del cnn_feats, zernike_feats
    batch_counter += 1
    
    print(f'Processed batch: {batch_counter} at: {time.perf_counter():.2f}')
