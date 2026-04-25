from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from PIL import Image

from dataset_utils import SampleRecord
from einar_busternet.dataset import BusterNetDataset


def _write_png(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


def _write_mask(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values.astype(np.uint8))


def _sample(root: Path, label: str, case_id: str) -> SampleRecord:
    image_path = root / "train_images" / label / f"{case_id}.png"
    mask_paths = (root / "train_masks" / f"{case_id}.npy",) if label == "forged" else tuple()
    return SampleRecord(
        sample_id=f"{label}:{case_id}",
        case_id=case_id,
        label=label,  # type: ignore[arg-type]
        image_path=image_path,
        mask_paths=mask_paths,
        group_id=case_id,
        split="train",
    )


def _write_metadata(root: Path) -> None:
    pd.DataFrame(
        [
            {"case_id": "10", "status": "derived_from_pair"},
            {"case_id": "20", "status": "target_only_no_authentic"},
        ]
    ).to_csv(root / "train_masks_source_target_metadata.csv", index=False)


class BusterNetDatasetTests(unittest.TestCase):
    def test_filters_to_derived_pairs_and_authentic_counterparts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_metadata(root)
            image = np.full((5, 7), 127, dtype=np.uint8)
            for label, case_id in (("forged", "10"), ("authentic", "10"), ("forged", "20"), ("authentic", "30")):
                _write_png(root / "train_images" / label / f"{case_id}.png", image)
            _write_mask(root / "train_masks_source" / "10.npy", np.zeros((5, 7), dtype=np.uint8))
            _write_mask(root / "train_masks_target" / "10.npy", np.ones((5, 7), dtype=np.uint8))

            samples = [_sample(root, "forged", "10"), _sample(root, "authentic", "10"), _sample(root, "forged", "20"), _sample(root, "authentic", "30")]

            dataset = BusterNetDataset(samples, data_root=root, target_size=8, normalize_rgb=False)

            self.assertEqual([sample.sample_id for sample in dataset.samples], ["forged:10", "authentic:10"])

    def test_paired_forged_sample_returns_three_class_long_label_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_metadata(root)
            _write_png(root / "train_images" / "forged" / "10.png", np.full((4, 4), 200, dtype=np.uint8))
            source = np.zeros((4, 4), dtype=np.uint8)
            target = np.zeros((4, 4), dtype=np.uint8)
            target[0:2, 0:2] = 1
            source[2:4, 2:4] = 1
            _write_mask(root / "train_masks_source" / "10.npy", source)
            _write_mask(root / "train_masks_target" / "10.npy", target)

            dataset = BusterNetDataset([_sample(root, "forged", "10")], data_root=root, target_size=8, normalize_rgb=False)
            image, label = dataset[0]

            self.assertEqual(tuple(image.shape), (3, 8, 8))
            self.assertEqual(tuple(label.shape), (8, 8))
            self.assertEqual(label.dtype, torch.long)
            self.assertEqual(set(label.unique().tolist()), {0, 1, 2})

    def test_authentic_sample_returns_all_background_label_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_metadata(root)
            _write_png(root / "train_images" / "authentic" / "10.png", np.full((5, 7), 127, dtype=np.uint8))

            dataset = BusterNetDataset([_sample(root, "authentic", "10")], data_root=root, target_size=8, normalize_rgb=False)
            image, label = dataset[0]

            self.assertEqual(tuple(image.shape), (3, 8, 8))
            self.assertEqual(tuple(label.shape), (8, 8))
            self.assertEqual(int(label.sum().item()), 0)

    def test_missing_step0_masks_fail_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_metadata(root)
            _write_png(root / "train_images" / "forged" / "10.png", np.full((5, 7), 127, dtype=np.uint8))

            with self.assertRaisesRegex(FileNotFoundError, "Run einar_busternet.generate_source_target_masks"):
                BusterNetDataset([_sample(root, "forged", "10")], data_root=root, target_size=8)

    def test_resized_label_map_keeps_integer_class_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_metadata(root)
            _write_png(root / "train_images" / "forged" / "10.png", np.full((3, 3), 127, dtype=np.uint8))
            source = np.zeros((3, 3), dtype=np.uint8)
            target = np.zeros((3, 3), dtype=np.uint8)
            target[0, 0] = 1
            source[2, 2] = 1
            _write_mask(root / "train_masks_source" / "10.npy", source)
            _write_mask(root / "train_masks_target" / "10.npy", target)

            dataset = BusterNetDataset([_sample(root, "forged", "10")], data_root=root, target_size=7, normalize_rgb=False)
            _image, label = dataset[0]

            self.assertLessEqual(set(label.unique().tolist()), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
