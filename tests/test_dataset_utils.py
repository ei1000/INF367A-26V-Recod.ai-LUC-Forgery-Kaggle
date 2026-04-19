from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from dataset_utils import (
    SampleRecord,
    find_image_path,
    list_labeled_samples,
    load_image_from_path,
    load_instance_masks,
    load_union_mask_from_paths,
)


def _write_png(path: Path, value: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 5), value, dtype=np.uint8)).save(path)


def _write_mask(path: Path, coords: list[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((4, 5), dtype=np.uint8)
    for row, col in coords:
        mask[row, col] = 1
    np.save(path, mask)


class DatasetUtilsTests(unittest.TestCase):
    def test_list_labeled_samples_keeps_forged_and_authentic_pairs_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_png(root / "train_images" / "forged" / "10.png")
            _write_png(root / "train_images" / "authentic" / "10.png")
            _write_mask(root / "train_masks" / "10.npy", [(1, 2)])

            samples = list_labeled_samples(root)

            self.assertEqual([sample.sample_id for sample in samples], ["authentic:10", "forged:10"])
            self.assertEqual({sample.group_id for sample in samples}, {"10"})
            forged = next(sample for sample in samples if sample.label == "forged")
            authentic = next(sample for sample in samples if sample.label == "authentic")
            self.assertEqual(len(forged.mask_paths), 1)
            self.assertEqual(authentic.mask_paths, tuple())
            self.assertEqual(forged.image_path, root / "train_images" / "forged" / "10.png")
            self.assertEqual(authentic.image_path, root / "train_images" / "authentic" / "10.png")

    def test_find_image_path_can_be_label_specific_when_stems_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_png(root / "train_images" / "forged" / "10.png", value=100)
            _write_png(root / "train_images" / "authentic" / "10.png", value=200)

            forged_path = find_image_path("10", label="forged", data_root=root)
            authentic_path = find_image_path("10", label="authentic", data_root=root)

            self.assertEqual(forged_path, root / "train_images" / "forged" / "10.png")
            self.assertEqual(authentic_path, root / "train_images" / "authentic" / "10.png")

    def test_load_helpers_use_paths_and_preserve_binary_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "train_images" / "forged" / "20.png"
            mask_a = root / "train_masks" / "20.npy"
            mask_b = root / "train_masks" / "20_1.npy"
            _write_png(image_path, value=150)
            _write_mask(mask_a, [(0, 0)])
            _write_mask(mask_b, [(3, 4)])

            image = load_image_from_path(image_path)
            instances = load_instance_masks((mask_a, mask_b))
            union = load_union_mask_from_paths((mask_a, mask_b))

            self.assertEqual(image.shape, (4, 5))
            self.assertEqual(len(instances), 2)
            self.assertEqual(int(instances[0].sum()), 1)
            self.assertEqual(int(instances[1].sum()), 1)
            self.assertEqual(int(union.sum()), 2)
            self.assertEqual(union.dtype, np.uint8)

    def test_sample_record_split_assignment_returns_new_record(self) -> None:
        sample = SampleRecord(
            sample_id="forged:10",
            case_id="10",
            label="forged",
            image_path=Path("data/train_images/forged/10.png"),
            mask_paths=(Path("data/train_masks/10.npy"),),
            group_id="10",
            split=None,
        )

        assigned = sample.with_split("train")

        self.assertEqual(assigned.split, "train")
        self.assertIsNone(sample.split)
        self.assertEqual(assigned.sample_id, sample.sample_id)


if __name__ == "__main__":
    unittest.main()
