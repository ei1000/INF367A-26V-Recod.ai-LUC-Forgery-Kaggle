import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from dataset import ForgeryDataset
from torch.utils.data import DataLoader
from torchvision.utils import draw_segmentation_masks
from torchvision.datasets import ImageFolder

'''
Display image, optionally with mask
'''
def display_image(img, mask=None):
    img = img.detach().cpu()


    if mask is not None:
       img = draw_segmentation_masks(
           img,
           mask.bool(),
           alpha=0.9,
           colors="blue"
       ) 

    img_np = img.permute(1, 2, 0).numpy()

    plt.imshow(img_np)
    plt.axis('off')
    plt.show()

'''
Display generated cnn and zernike offsets as heatmaps
Optionally with original image as well
Shows dx and dy separately (like the paper)
'''
def display_pixel_offsets(cnn_offsets, zernike_offsets, img=None):
    def split_offsets(offsets):
        offsets = offsets.detach().cpu()
        if offsets.ndim == 3 and offsets.shape[0] == 2:
            dx = offsets[0].numpy()
            dy = offsets[1].numpy()
        elif offsets.ndim == 3 and offsets.shape[-1] == 2:
            dx = offsets[..., 0].numpy()
            dy = offsets[..., 1].numpy()
        else:
            raise ValueError(f"Offsets must be (2,H,W) or (H,W,2); got {offsets.shape}")
        return dx, dy

    cnn_dx, cnn_dy = split_offsets(cnn_offsets)
    zernike_dx, zernike_dy = split_offsets(zernike_offsets)

    cnn_vmax = max(float(np.abs(cnn_dx).max()), float(np.abs(cnn_dy).max()), 1e-6)
    z_vmax = max(float(np.abs(zernike_dx).max()), float(np.abs(zernike_dy).max()), 1e-6)

    ncols = 5 if img is not None else 4
    fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))

    axs[0].imshow(cnn_dx, cmap='coolwarm', vmin=-cnn_vmax, vmax=cnn_vmax)
    axs[0].set_title("CNN dx")
    axs[0].axis('off')

    axs[1].imshow(cnn_dy, cmap='coolwarm', vmin=-cnn_vmax, vmax=cnn_vmax)
    axs[1].set_title("CNN dy")
    axs[1].axis('off')

    axs[2].imshow(zernike_dx, cmap='coolwarm', vmin=-z_vmax, vmax=z_vmax)
    axs[2].set_title("Zernike dx")
    axs[2].axis('off')

    axs[3].imshow(zernike_dy, cmap='coolwarm', vmin=-z_vmax, vmax=z_vmax)
    axs[3].set_title("Zernike dy")
    axs[3].axis('off')

    if img is not None:
        img = img.detach().cpu()
        img_np = img.permute(1, 2, 0).numpy()
        axs[4].imshow(img_np)
        axs[4].set_title("Image")
        axs[4].axis('off')

    fig.suptitle("Image offsets (dx/dy)")
    plt.tight_layout()
    plt.show()
# Test:
def load_and_display():
    root = Path('data')
    supplement_image_folder = ImageFolder(root / "supplemental_images")

    samples = [(Path(p), y) for p, y in supplement_image_folder.samples]

    supplement_dataset = ForgeryDataset(
        samples=samples,
        mask_dir=root / "supplemental_masks",
    )

    img, mask, _ = supplement_dataset[1]
    display_image(img, mask)

