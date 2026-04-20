from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from dataset_utils import SampleRecord
import datasets.forgery_dataset as forgery_dataset_module
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
            forged_image = root / "train_images" / "forged" / "sample_a.png"
            decoy_image = root / "train_images" / "forged" / "10.png"
            forged_mask = root / "train_masks" / "sample_a.npy"
            decoy_mask = root / "train_masks" / "10.npy"
            _write_png(forged_image, np.full((5, 7), 255, dtype=np.uint8))
            _write_png(decoy_image, np.zeros((5, 7), dtype=np.uint8))
            mask_values = np.zeros((5, 7), dtype=np.uint8)
            mask_values[2, 3] = 1
            _write_mask(forged_mask, mask_values)
            _write_mask(decoy_mask, np.zeros((5, 7), dtype=np.uint8))
            sample = SampleRecord(
                sample_id="forged:sample_a",
                case_id="10",
                label="forged",
                image_path=forged_image,
                mask_paths=(forged_mask,),
                group_id="sample_a",
                split="train",
            )

            dataset = ForgeryDataset([sample], target_size=8, use_rgb=False)
            image, mask = dataset[0]

            self.assertEqual(tuple(image.shape), (1, 8, 8))
            self.assertEqual(tuple(mask.shape), (1, 8, 8))
            self.assertGreater(float(image.mean().item()), 0.9)
            self.assertGreater(float(mask.sum().item()), 0.0)

    def test_string_case_id_still_uses_legacy_helpers(self) -> None:
        original_load_image = forgery_dataset_module.load_image
        original_load_union_mask = forgery_dataset_module.load_union_mask
        try:
            forged_image = np.full((5, 7), 255, dtype=np.float32)
            forged_mask = np.zeros((5, 7), dtype=np.uint8)
            forged_mask[1, 2] = 1

            forgery_dataset_module.load_image = lambda case_id: forged_image if case_id == "10" else None
            forgery_dataset_module.load_union_mask = lambda case_id: forged_mask if case_id == "10" else None

            dataset = ForgeryDataset(["10"], target_size=8, use_rgb=False)
            image, mask = dataset[0]

            self.assertEqual(tuple(image.shape), (1, 8, 8))
            self.assertEqual(tuple(mask.shape), (1, 8, 8))
            self.assertGreater(float(image.mean().item()), 0.9)
            self.assertGreater(float(mask.sum().item()), 0.0)
        finally:
            forgery_dataset_module.load_image = original_load_image
            forgery_dataset_module.load_union_mask = original_load_union_mask


if __name__ == "__main__":
    unittest.main()
