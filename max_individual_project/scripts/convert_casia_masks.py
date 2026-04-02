import numpy as np
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import resolve_data_root


data_root = resolve_data_root()
input_dir = data_root / "casia_cmfd_masks"
output_dir = data_root / "casia_cmfd_masks_np"

for mask_path in input_dir.glob("*.png"):
    mask_img = np.array(Image.open(mask_path).convert("RGB"))

    # Split channels
    r = mask_img[:, :, 0]
    g = mask_img[:, :, 1]
    b = mask_img[:, :, 2]

    # Detect blue background
    is_blue = (b > 200) & (r < 50) & (g < 50)

    # Everything NOT blue = forged (1)
    binary_mask = (~is_blue).astype(np.uint8)

    np.save(output_dir / f"{mask_path.stem}.npy", binary_mask)