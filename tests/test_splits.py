from pathlib import Path
import unittest

from dataset_utils import SampleRecord
from datasets.splits import count_samples_by_split_and_label, make_grouped_stratified_splits


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
        split=None,
    )


class SplitTests(unittest.TestCase):
    def test_grouped_split_keeps_paired_stems_together(self) -> None:
        samples = []
        for idx in range(10):
            case_id = str(idx)
            samples.append(_sample("forged", case_id))
            samples.append(_sample("authentic", case_id))

        splits = make_grouped_stratified_splits(
            samples,
            seed=123,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )

        group_to_split: dict[str, str] = {}
        for split_name, split_samples in splits.items():
            for sample in split_samples:
                previous = group_to_split.setdefault(sample.group_id, split_name)
                self.assertEqual(previous, split_name)
                self.assertEqual(sample.split, split_name)

        self.assertEqual(set(splits), {"train", "val", "test"})
        self.assertEqual(sum(len(value) for value in splits.values()), 20)

    def test_split_is_seeded_and_deterministic(self) -> None:
        samples = [_sample("forged", str(idx)) for idx in range(25)]

        first = make_grouped_stratified_splits(samples, seed=42)
        second = make_grouped_stratified_splits(samples, seed=42)
        third = make_grouped_stratified_splits(samples, seed=43)

        self.assertEqual(
            [[sample.sample_id for sample in first[name]] for name in ("train", "val", "test")],
            [[sample.sample_id for sample in second[name]] for name in ("train", "val", "test")],
        )
        self.assertNotEqual(
            [sample.sample_id for sample in first["train"]],
            [sample.sample_id for sample in third["train"]],
        )

    def test_counts_report_split_and_label_totals(self) -> None:
        samples = []
        for idx in range(10):
            case_id = str(idx)
            samples.append(_sample("forged", case_id))
            samples.append(_sample("authentic", case_id))

        splits = make_grouped_stratified_splits(samples, seed=1)
        counts = count_samples_by_split_and_label(splits)

        self.assertEqual(counts["total"], 20)
        self.assertEqual(counts["by_label"]["forged"], 10)
        self.assertEqual(counts["by_label"]["authentic"], 10)
        self.assertEqual(counts["by_split"]["train"]["total"], 16)
        self.assertEqual(counts["by_split"]["val"]["total"], 2)
        self.assertEqual(counts["by_split"]["test"]["total"], 2)


if __name__ == "__main__":
    unittest.main()
