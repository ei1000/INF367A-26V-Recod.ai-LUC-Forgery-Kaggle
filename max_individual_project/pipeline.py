from pathlib import Path
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dataset import ForgeryDataset, Datasets, regular_transform
import time

from feature_extractors.cnn_feature_extractor import PyramidFeatureExtractor
from feature_extractors.zernike_feature_extractor import PyramidZernikeExtractor
from cross_scale_patternmatch.pixel_propagator import PixelPropagator

from visualizer import *

def pipeline(datasets=Datasets.TRAIN, image_size=488, test_run=False):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    root = Path('data')

    for dataset in datasets.value:
        image_folder = ImageFolder(root / dataset['images'])

        samples = [(Path(p), y) for p, y in image_folder.samples]

        forgery_dataset = ForgeryDataset(
            samples=samples,
            mask_dir=root / dataset['masks'] if dataset['masks'] is not None else None,
            size=image_size,
            transform=regular_transform
        )

    train_loader = DataLoader(forgery_dataset, batch_size=8, shuffle=True)

    # Feature extraction
    pyramid_bb = PyramidFeatureExtractor().to(device)

    # Zernike pair things
    pq_list = [(0,0),(1,1),(0,2),(1,3),(0,4),(1,5)]
    pyramid_zm = PyramidZernikeExtractor(pq_list, kernel_size=64).to(device)

    # MAIN LOOP
    batch_counter = 0

    for images, masks, labels in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            cnn_feats = pyramid_bb(images)
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
