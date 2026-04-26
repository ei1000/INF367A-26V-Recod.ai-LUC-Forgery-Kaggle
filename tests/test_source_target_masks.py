import unittest

import numpy as np

from einar_busternet.source_target_masks import derive_source_target_masks_from_arrays


class SourceTargetMasksTests(unittest.TestCase):
    def test_uses_difference_to_classify_clean_union_components(self) -> None:
        authentic = np.zeros((6, 8), dtype=np.uint8)
        forged = authentic.copy()
        forged[1, 1] = 4
        forged[1:3, 5:7] = 100
        union_mask = np.zeros((6, 8), dtype=np.uint8)
        union_mask[1:3, 1:3] = 1
        union_mask[1:3, 5:7] = 1

        masks = derive_source_target_masks_from_arrays(
            authentic,
            forged,
            union_mask,
            diff_threshold=5.0,
            component_change_fraction=0.25,
        )

        self.assertEqual(masks.status, "derived_from_pair")
        self.assertEqual(int(masks.union_mask.sum()), 8)
        self.assertEqual(int(masks.source_mask.sum()), 4)
        self.assertEqual(int(masks.target_mask.sum()), 4)

    def test_without_authentic_pair_returns_target_only(self) -> None:
        forged = np.zeros((6, 8), dtype=np.uint8)
        union_mask = np.zeros((6, 8), dtype=np.uint8)
        union_mask[1:3, 1:3] = 1

        masks = derive_source_target_masks_from_arrays(None, forged, union_mask)

        self.assertEqual(masks.status, "target_only_no_authentic")
        self.assertEqual(int(masks.source_mask.sum()), 0)
        self.assertEqual(int(masks.target_mask.sum()), 4)

    def test_pair_with_only_faint_difference_still_gets_target_component(self) -> None:
        authentic = np.zeros((6, 8), dtype=np.uint8)
        forged = authentic.copy()
        forged[1:3, 5:7] = 4
        union_mask = np.zeros((6, 8), dtype=np.uint8)
        union_mask[1:3, 1:3] = 1
        union_mask[1:3, 5:7] = 1

        masks = derive_source_target_masks_from_arrays(
            authentic,
            forged,
            union_mask,
            diff_threshold=5.0,
            component_change_fraction=0.25,
        )

        self.assertEqual(int(masks.target_mask.sum()), 4)
        self.assertEqual(int(masks.source_mask.sum()), 4)


if __name__ == "__main__":
    unittest.main()
