from pathlib import Path
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from dataset import ForgeryDataset


from feature_extractors.cnn_feature_extractor import PyramidFeatureExtractor
from feature_extractors.zernike_feature_extractor import PyramidZernikeExtractor
from cross_scale_patternmatch.pixel_propagator import propagation_layer


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
'''
pq_list = [(0,0),(1,1),(0,2),(1,3),(0,4),(1,5)]
pyramid_zm = PyramidZernikeExtractor(pq_list, kernel_size=64).to(device)
'''

for images, _, _ in train_loader:
    images = images.to(device)
    for img in images:
        grid = propagation_layer(img)
        print(grid.shape)
        print(grid)
        break
    break

    '''
    Fcb, Fco, Fcu = pyramid_bb(images)
    Fzb, Fzo, Fzu = pyramid_zm(images)
    break
    '''

'''
print(Fcb)
print(Fco)
print(Fcu)


print(Fzb)
print(Fzo)
print(Fzu)
'''