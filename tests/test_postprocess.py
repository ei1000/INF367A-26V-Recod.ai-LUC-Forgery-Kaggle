import unittest

import numpy as np
from scipy import ndimage

from util.pixelmapUtil import PixelMapUtil


class PostProcessKnobTests(unittest.TestCase):
    def test_smooth_probabilities_false_changes_behavior(self) -> None:
        probs = np.zeros((21, 21), dtype=np.float32)
        probs[9:12, 9:12] = 1.0
        pixel_util = PixelMapUtil(gaussian_sigma=2.0)

        result_smooth = pixel_util.post_process_mask_probs(
            probs,
            threshold=0.1,
            confident_threshold=0.95,
            smooth_probabilities=True,
            fill_holes=False,
            apply_opening=False,
            apply_closing=False,
        )
        result_raw = pixel_util.post_process_mask_probs(
            probs,
            threshold=0.1,
            confident_threshold=0.95,
            smooth_probabilities=False,
            fill_holes=False,
            apply_opening=False,
            apply_closing=False,
        )

        self.assertNotEqual(float(result_smooth.sum()), float(result_raw.sum()))
        self.assertGreater(float(result_smooth.sum()), float(result_raw.sum()))

    def test_fill_holes_true_fills_interior_void(self) -> None:
        probs = np.zeros((12, 12), dtype=np.float32)
        probs[2:10, 2:10] = 1.0
        probs[4:8, 4:8] = 0.0
        pixel_util = PixelMapUtil()

        result_fill = pixel_util.post_process_mask_probs(
            probs,
            threshold=0.5,
            confident_threshold=0.95,
            smooth_probabilities=False,
            fill_holes=True,
            apply_opening=False,
            apply_closing=False,
        )
        result_nofill = pixel_util.post_process_mask_probs(
            probs,
            threshold=0.5,
            confident_threshold=0.95,
            smooth_probabilities=False,
            fill_holes=False,
            apply_opening=False,
            apply_closing=False,
        )

        self.assertGreater(float(result_fill.sum()), float(result_nofill.sum()))

    def test_apply_opening_false_preserves_isolated_pixel(self) -> None:
        probs = np.zeros((15, 15), dtype=np.float32)
        probs[5:10, 5:10] = 1.0
        probs[1, 1] = 1.0
        pixel_util = PixelMapUtil()

        result_open = pixel_util.post_process_mask_probs(
            probs,
            threshold=0.5,
            confident_threshold=0.95,
            smooth_probabilities=False,
            fill_holes=False,
            apply_opening=True,
            apply_closing=False,
        )
        result_noopen = pixel_util.post_process_mask_probs(
            probs,
            threshold=0.5,
            confident_threshold=0.95,
            smooth_probabilities=False,
            fill_holes=False,
            apply_opening=False,
            apply_closing=False,
        )

        self.assertGreater(float(result_noopen.sum()), float(result_open.sum()))

    def test_apply_closing_true_connects_nearby_components(self) -> None:
        probs = np.zeros((16, 16), dtype=np.float32)
        probs[3:6, 3:6] = 1.0
        probs[3:6, 8:11] = 1.0
        pixel_util = PixelMapUtil()

        result_close = pixel_util.post_process_mask_probs(
            probs,
            threshold=0.5,
            confident_threshold=0.95,
            smooth_probabilities=False,
            fill_holes=False,
            apply_opening=False,
            apply_closing=True,
        )
        result_noclose = pixel_util.post_process_mask_probs(
            probs,
            threshold=0.5,
            confident_threshold=0.95,
            smooth_probabilities=False,
            fill_holes=False,
            apply_opening=False,
            apply_closing=False,
        )

        close_components = ndimage.label(result_close.astype(bool))[1]
        noclose_components = ndimage.label(result_noclose.astype(bool))[1]
        self.assertLess(close_components, noclose_components)

    def test_confident_threshold_controls_confident_seeding(self) -> None:
        probs = np.zeros((10, 10), dtype=np.float32)
        probs[5, 5] = 0.7
        pixel_util = PixelMapUtil()

        result_low_conf = pixel_util.post_process_mask_probs(
            probs,
            threshold=0.8,
            confident_threshold=0.6,
            smooth_probabilities=False,
            fill_holes=False,
            apply_opening=False,
            apply_closing=False,
        )
        result_high_conf = pixel_util.post_process_mask_probs(
            probs,
            threshold=0.8,
            confident_threshold=0.9,
            smooth_probabilities=False,
            fill_holes=False,
            apply_opening=False,
            apply_closing=False,
        )

        self.assertGreater(float(result_low_conf.sum()), float(result_high_conf.sum()))

    def test_confident_threshold_none_behaves_like_default(self) -> None:
        probs = np.zeros((10, 10), dtype=np.float32)
        probs[5, 5] = 0.95
        pixel_util = PixelMapUtil()

        result_none = pixel_util.post_process_mask_probs(
            probs,
            threshold=0.8,
            confident_threshold=None,
            smooth_probabilities=False,
            fill_holes=False,
            apply_opening=False,
            apply_closing=False,
        )
        result_default = pixel_util.post_process_mask_probs(
            probs,
            threshold=0.8,
            confident_threshold=0.9,
            smooth_probabilities=False,
            fill_holes=False,
            apply_opening=False,
            apply_closing=False,
        )

        np.testing.assert_array_equal(result_none, result_default)

    def test_keep_confident_seeded_components_removes_unseeded_components(self) -> None:
        probs = np.zeros((20, 20), dtype=np.float32)
        probs[2:6, 2:6] = 0.55
        probs[14:18, 14:18] = 0.95
        pixel_util = PixelMapUtil()

        result_keep = pixel_util.post_process_mask_probs(
            probs,
            threshold=0.5,
            confident_threshold=0.9,
            smooth_probabilities=False,
            fill_holes=False,
            apply_opening=False,
            apply_closing=False,
            keep_confident_seeded_components=True,
        )
        result_nokeep = pixel_util.post_process_mask_probs(
            probs,
            threshold=0.5,
            confident_threshold=0.9,
            smooth_probabilities=False,
            fill_holes=False,
            apply_opening=False,
            apply_closing=False,
            keep_confident_seeded_components=False,
        )

        self.assertGreater(float(result_nokeep.sum()), float(result_keep.sum()))
        self.assertEqual(float(result_keep.sum()), 16.0)


if __name__ == "__main__":
    unittest.main()
