import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from dataset import ForgeryDataset
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
def display_pixel_offsets(
    cnn_offsets,
    zernike_offsets,
    img=None,
    clip_percentile=99.0,
    scale=1.0,
    cmap='magma',
    show_colorbar=False,
):
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

    def sym_vmax(dx, dy, p):
        vals = np.concatenate([np.abs(dx).reshape(-1), np.abs(dy).reshape(-1)])
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            return 1.0
        vmax = float(np.percentile(vals, p))
        return max(vmax, 1e-6)

    cnn_dx, cnn_dy = split_offsets(cnn_offsets)
    zernike_dx, zernike_dy = split_offsets(zernike_offsets)

    if scale != 1.0:
        cnn_dx = cnn_dx * scale
        cnn_dy = cnn_dy * scale
        zernike_dx = zernike_dx * scale
        zernike_dy = zernike_dy * scale

    cnn_vmax = sym_vmax(cnn_dx, cnn_dy, clip_percentile)
    z_vmax = sym_vmax(zernike_dx, zernike_dy, clip_percentile)

    ncols = 5 if img is not None else 4
    fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))

    im0 = axs[0].imshow(cnn_dx, cmap=cmap, vmin=-cnn_vmax, vmax=cnn_vmax)
    axs[0].set_title("CNN dx")
    axs[0].axis('off')

    im1 = axs[1].imshow(cnn_dy, cmap=cmap, vmin=-cnn_vmax, vmax=cnn_vmax)
    axs[1].set_title("CNN dy")
    axs[1].axis('off')

    im2 = axs[2].imshow(zernike_dx, cmap=cmap, vmin=-z_vmax, vmax=z_vmax)
    axs[2].set_title("Zernike dx")
    axs[2].axis('off')

    im3 = axs[3].imshow(zernike_dy, cmap=cmap, vmin=-z_vmax, vmax=z_vmax)
    axs[3].set_title("Zernike dy")
    axs[3].axis('off')

    if show_colorbar:
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
        fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

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

