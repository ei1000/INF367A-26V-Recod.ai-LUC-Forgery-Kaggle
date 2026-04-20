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
    load_resized_instance_masks,
    mask_instances_to_annotation,
    score_instances,
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
        self.assertEqual(score_instances([], []), 1.0)
        self.assertEqual(score_instances([_mask([(1, 1)])], []), 0.0)
        self.assertEqual(score_instances([], [_mask([(1, 1)])]), 0.0)

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

    def test_load_resized_instance_masks_rejects_forged_sample_without_masks(self) -> None:
        forged = SampleRecord(
            sample_id="forged:11",
            case_id="11",
            label="forged",
            image_path=Path("data/train_images/forged/11.png"),
            mask_paths=tuple(),
            group_id="11",
            split="val",
        )

        with self.assertRaisesRegex(ValueError, "forged:11.*11.*missing.*mask"):
            load_resized_instance_masks(forged, (6, 6))

    def test_missing_prediction_defaults_to_authentic_output(self) -> None:
        authentic = _sample("authentic", "10")

        score = compute_kaggle_score_from_instances(
            ordered_samples=[authentic],
            pred_instances_by_sample_id={},
            gt_instances_by_sample_id={"authentic:10": []},
        )

        submission = build_submission_rows([authentic], {})

        self.assertEqual(score, 1.0)
        self.assertEqual(submission.loc[0, "annotation"], "authentic")

    def test_forged_missing_gt_mapping_raises_clear_error(self) -> None:
        forged = _sample("forged", "11")

        with self.assertRaisesRegex(KeyError, "forged:11"):
            compute_kaggle_score_from_instances(
                ordered_samples=[forged],
                pred_instances_by_sample_id={"forged:11": [_mask([(1, 1)])]},
                gt_instances_by_sample_id={},
            )

    def test_forged_empty_gt_instances_raise_clear_error(self) -> None:
        forged = _sample("forged", "11")

        with self.assertRaisesRegex(ValueError, "forged:11"):
            compute_kaggle_score_from_instances(
                ordered_samples=[forged],
                pred_instances_by_sample_id={"forged:11": [_mask([(1, 1)])]},
                gt_instances_by_sample_id={"forged:11": []},
            )

        with self.assertRaisesRegex(ValueError, "forged:11"):
            build_solution_rows(
                ordered_samples=[forged],
                gt_instances_by_sample_id={"forged:11": []},
                shapes_by_sample_id={"forged:11": (6, 6)},
            )

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

        self.assertEqual(solution.loc[0, "annotation"], "authentic")
        self.assertEqual(solution.loc[0, "shape"], "authentic")
        self.assertEqual(solution.loc[1, "shape"], "[6, 6]")
        self.assertEqual(submission.loc[0, "annotation"], "authentic")
        self.assertEqual(mask_instances_to_annotation([]), "authentic")
        self.assertAlmostEqual(direct, official, places=8)
        self.assertAlmostEqual(official, 1.0, places=8)

    def test_official_parity_penalizes_excess_predictions(self) -> None:
        forged = _sample("forged", "11")
        ordered_samples = [forged]
        gt_instances = {"forged:11": [_mask([(1, 1), (1, 2)])]}
        pred_instances = {
            "forged:11": [
                _mask([(1, 1), (1, 2)]),
                _mask([(4, 4)]),
            ]
        }
        shapes = {"forged:11": (6, 6)}

        solution = build_solution_rows(ordered_samples, gt_instances, shapes)
        submission = build_submission_rows(ordered_samples, pred_instances)
        direct = compute_kaggle_score_from_instances(ordered_samples, pred_instances, gt_instances)
        official = compute_kaggle_score_via_recodai(solution, submission)

        self.assertAlmostEqual(direct, official, places=8)
        self.assertLess(direct, 1.0)

    def test_official_parity_handles_partial_overlap(self) -> None:
        forged = _sample("forged", "11")
        ordered_samples = [forged]
        gt_instances = {"forged:11": [_mask([(1, 1), (1, 2)])]}
        pred_instances = {"forged:11": [_mask([(1, 1), (2, 2)])]}
        shapes = {"forged:11": (6, 6)}

        solution = build_solution_rows(ordered_samples, gt_instances, shapes)
        submission = build_submission_rows(ordered_samples, pred_instances)
        direct = compute_kaggle_score_from_instances(ordered_samples, pred_instances, gt_instances)
        official = compute_kaggle_score_via_recodai(solution, submission)

        self.assertAlmostEqual(direct, official, places=8)
        self.assertLess(direct, 1.0)


if __name__ == "__main__":
    unittest.main()
