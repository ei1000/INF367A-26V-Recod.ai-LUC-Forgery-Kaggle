import matplotlib.pyplot as plt
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
'''
def display_pixel_offsets(cnn_offsets, zernike_offsets, img=None):
    ncols = 3 if img is not None else 2
    fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))

    def to_heatmap(offsets):
        offsets = offsets.detach().cpu()
        if offsets.ndim == 3 and offsets.shape[0] in (2, 3):
            return torch.linalg.norm(offsets.float(), dim=0).numpy()
        return offsets.squeeze().numpy()

    cnn_offsets_np = to_heatmap(cnn_offsets)
    zernike_offsets_np = to_heatmap(zernike_offsets)

    axs[0].imshow(cnn_offsets_np, cmap='viridis')
    axs[0].set_title("CNN Offsets")

    axs[1].imshow(zernike_offsets_np, cmap='viridis')
    axs[1].set_title("Zernike Offsets")

    if img is not None:
        img = img.detach().cpu()
        img_np = img.permute(1, 2, 0).numpy()
        axs[2].imshow(img_np)
        axs[2].set_title("Image")

    fig.suptitle("Image offsets")
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



