from pathlib import Path
import unittest

import numpy as np

from dataset_utils import SampleRecord
from engine.validation_records import (
    build_solution_rows,
    build_submission_rows,
    compute_kaggle_score_from_instances,
    compute_kaggle_score_via_recodai,
    connected_components_to_masks,
    mask_instances_to_annotation,
)


def _sample(label: str, case_id: str) -> SampleRecord:
    image_dir = "forged" if label == "forged" else "authentic"
    masks = (Path(f"data/train_masks/{case_id}.npy"),) if label == "forged" else tuple()
    return SampleRecord(
        sample_id=f"{label}:{case_id}",
        case_id=case_id,
        label=label,
        image_path=Path(f"data/train_images/{image_dir}/{case_id}.png"),
        mask_paths=masks,
        group_id=case_id,
        split="val",
    )


def _mask(coords: list[tuple[int, int]], shape: tuple[int, int] = (6, 6)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for row, col in coords:
        mask[row, col] = 1
    return mask


class ValidationRecordTests(unittest.TestCase):
    def test_connected_components_split_prediction_instances(self) -> None:
        mask = _mask([(0, 0), (0, 1), (5, 5)])

        components = connected_components_to_masks(mask)

        self.assertEqual(len(components), 2)
        self.assertEqual(sorted(int(component.sum()) for component in components), [1, 2])

    def test_direct_score_handles_authentic_exact_match_semantics(self) -> None:
        authentic = _sample("authentic", "10")
        forged = _sample("forged", "11")
        gt = {
            "authentic:10": [],
            "forged:11": [_mask([(1, 1), (1, 2)])],
        }

        score = compute_kaggle_score_from_instances(
            ordered_samples=[authentic, forged],
            pred_instances_by_sample_id={
                "authentic:10": [],
                "forged:11": [_mask([(1, 1), (1, 2)])],
            },
            gt_instances_by_sample_id=gt,
        )

        self.assertEqual(score, 1.0)

    def test_direct_score_penalizes_false_positive_on_authentic(self) -> None:
        authentic = _sample("authentic", "10")

        score = compute_kaggle_score_from_instances(
            ordered_samples=[authentic],
            pred_instances_by_sample_id={"authentic:10": [_mask([(1, 1)])]},
            gt_instances_by_sample_id={"authentic:10": []},
        )

        self.assertEqual(score, 0.0)

    def test_direct_score_penalizes_empty_prediction_on_forged(self) -> None:
        forged = _sample("forged", "11")

        score = compute_kaggle_score_from_instances(
            ordered_samples=[forged],
            pred_instances_by_sample_id={"forged:11": []},
            gt_instances_by_sample_id={"forged:11": [_mask([(1, 1)])]},
        )

        self.assertEqual(score, 0.0)

    def test_annotation_rows_match_recodai_score_for_representative_cases(self) -> None:
        authentic = _sample("authentic", "10")
        forged = _sample("forged", "11")
        ordered_samples = [authentic, forged]
        gt_instances = {
            "authentic:10": [],
            "forged:11": [_mask([(1, 1), (1, 2)])],
        }
        pred_instances = {
            "authentic:10": [],
            "forged:11": [_mask([(1, 1), (1, 2)])],
        }
        shapes = {"authentic:10": (6, 6), "forged:11": (6, 6)}
        solution = build_solution_rows(ordered_samples, gt_instances, shapes)
        submission = build_submission_rows(ordered_samples, pred_instances)

        direct = compute_kaggle_score_from_instances(ordered_samples, pred_instances, gt_instances)
        official = compute_kaggle_score_via_recodai(solution, submission)

        self.assertEqual(mask_instances_to_annotation([]), "authentic")
        self.assertAlmostEqual(direct, official, places=8)
        self.assertAlmostEqual(official, 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
