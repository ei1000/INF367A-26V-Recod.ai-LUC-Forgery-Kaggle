from pathlib import Path
import tempfile
import unittest

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")

from visualization import ForgeryDataPlotter


def _write_png(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


def _write_mask(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values.astype(np.uint8))


class ForgeryDataPlotterTests(unittest.TestCase):
    def test_resolves_pair_and_plots_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_png(root / "train_images" / "authentic" / "10.png", np.zeros((6, 8), dtype=np.uint8))
            _write_png(root / "train_images" / "forged" / "10.png", np.full((6, 8), 100, dtype=np.uint8))
            mask = np.zeros((6, 8), dtype=np.uint8)
            mask[2:4, 3:5] = 1
            _write_mask(root / "train_masks" / "10.npy", mask)

            plotter = ForgeryDataPlotter(root)

            self.assertEqual(plotter.list_case_ids(require_authentic=True), ["10"])
            summary = plotter.summarize_cases(["10"])
            self.assertEqual(int(summary.loc[0, "mask_pixels"]), 4)

            fig, axes = plotter.plot_case("10")

            self.assertEqual(len(axes), 4)
            fig.clear()

    def test_derives_clean_source_target_components_from_pair_difference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            authentic = np.zeros((6, 8), dtype=np.uint8)
            forged = authentic.copy()
            forged[1, 1] = 40
            forged[1:3, 5:7] = 200
            union_mask = np.zeros((6, 8), dtype=np.uint8)
            union_mask[1:3, 1:3] = 1
            union_mask[1:3, 5:7] = 1

            _write_png(root / "train_images" / "authentic" / "20.png", authentic)
            _write_png(root / "train_images" / "forged" / "20.png", forged)
            _write_mask(root / "train_masks" / "20.npy", union_mask)

            plotter = ForgeryDataPlotter(root)
            masks = plotter.derive_source_target_masks("20", component_change_fraction=0.5)

            self.assertEqual(int(masks.union_mask.sum()), 8)
            self.assertEqual(int(masks.target_mask.sum()), 4)
            self.assertEqual(int(masks.source_mask.sum()), 4)
            self.assertTrue(np.array_equal(masks.union_mask, masks.source_mask | masks.target_mask))
            self.assertEqual(masks.component_scores.shape[0], 2)

            fig, axes = plotter.plot_source_target_split("20", component_change_fraction=0.5)

            self.assertEqual(len(axes), 5)
            fig.clear()

    def test_pixel_strategy_keeps_speckled_difference_for_debugging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            authentic = np.zeros((6, 8), dtype=np.uint8)
            forged = authentic.copy()
            forged[1, 1] = 40
            forged[1:3, 5:7] = 200
            union_mask = np.zeros((6, 8), dtype=np.uint8)
            union_mask[1:3, 1:3] = 1
            union_mask[1:3, 5:7] = 1

            _write_png(root / "train_images" / "authentic" / "21.png", authentic)
            _write_png(root / "train_images" / "forged" / "21.png", forged)
            _write_mask(root / "train_masks" / "21.npy", union_mask)

            plotter = ForgeryDataPlotter(root)
            masks = plotter.derive_source_target_masks("21", split_strategy="pixel")

            self.assertEqual(int(masks.target_mask.sum()), 5)
            self.assertEqual(int(masks.source_mask.sum()), 3)


if __name__ == "__main__":
    unittest.main()
