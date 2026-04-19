from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from dataset_utils import SampleRecord
from datasets.forgery_dataset import ForgeryDataset


def _write_png(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


def _write_mask(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values.astype(np.uint8))


class ForgeryDatasetTests(unittest.TestCase):
    def test_authentic_sample_returns_zero_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "train_images" / "authentic" / "10.png"
            _write_png(image_path, np.full((5, 7), 127, dtype=np.uint8))
            sample = SampleRecord(
                sample_id="authentic:10",
                case_id="10",
                label="authentic",
                image_path=image_path,
                mask_paths=tuple(),
                group_id="10",
                split="train",
            )

            dataset = ForgeryDataset([sample], target_size=8, use_rgb=True, normalize_rgb=False)
            image, mask = dataset[0]

            self.assertEqual(tuple(image.shape), (3, 8, 8))
            self.assertEqual(tuple(mask.shape), (1, 8, 8))
            self.assertEqual(float(mask.sum().item()), 0.0)

    def test_forged_sample_uses_sample_image_path_and_union_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forged_image = root / "train_images" / "forged" / "10.png"
            authentic_image = root / "train_images" / "authentic" / "10.png"
            mask_path = root / "train_masks" / "10.npy"
            _write_png(forged_image, np.full((5, 7), 255, dtype=np.uint8))
            _write_png(authentic_image, np.zeros((5, 7), dtype=np.uint8))
            mask_values = np.zeros((5, 7), dtype=np.uint8)
            mask_values[2, 3] = 1
            _write_mask(mask_path, mask_values)
            sample = SampleRecord(
                sample_id="forged:10",
                case_id="10",
                label="forged",
                image_path=forged_image,
                mask_paths=(mask_path,),
                group_id="10",
                split="train",
            )

            dataset = ForgeryDataset([sample], target_size=8, use_rgb=False)
            image, mask = dataset[0]

            self.assertEqual(tuple(image.shape), (1, 8, 8))
            self.assertEqual(tuple(mask.shape), (1, 8, 8))
            self.assertGreater(float(image.mean().item()), 0.9)
            self.assertGreater(float(mask.sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
