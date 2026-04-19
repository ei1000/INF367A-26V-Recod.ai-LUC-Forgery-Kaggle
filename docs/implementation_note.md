# Baseline Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. In this repository, do not start implementation until the user gives a direct implementation signal.

**Goal:** Refactor the DINOv2 baseline so training uses all labeled data, validation uses Kaggle-equivalent instance scoring, authentic images are handled correctly, and model selection is based on `kaggle_score`.

**Architecture:** Keep the current training stack and DINOv2 model structure. Add explicit sample metadata, grouped deterministic splits, authentic zero-mask support, and validation helpers that compare a direct fast scorer against `recodai_f1.score`.

**Tech Stack:** Python 3.12, PyTorch, NumPy, Pillow, SciPy, pandas, numba, stdlib `unittest`, existing `recodai_f1.py`.

---

## Scope Check

This plan covers the core metric/data correctness refactor from `docs/baseline_refactor_spec.md` and `docs/baseline_refactor_plan.md`.

In scope:

- sample metadata and labeled data inventory,
- grouped 80/10/10 train/validation/local-test split,
- authentic samples with all-zero training masks,
- forged instance-mask preservation for validation scoring,
- direct Kaggle-equivalent scorer verified against `recodai_f1.score`,
- validation result dictionary with `kaggle_score`,
- checkpoint selection by `kaggle_score`,
- opt-in `pixel_f1`,
- clear deprecation markers for `main.py` and `main.ipynb`.

Out of scope for this note:

- DINO architecture changes,
- Hugging Face model loading for training,
- TTA, global/local fusion, or notebook-style final inference entrypoint,
- README and pipeline documentation rewrite,
- threshold tuning or leaderboard optimization,
- using supplemental data for training or validation.

## Commit Policy

The steps include commit checkpoints because the planning prompt asks for frequent commits. In this repo, only run the `git commit` commands if the user explicitly approves commits during implementation. Otherwise, use those steps as review checkpoints.

## Human Checkpoints

Most tasks are pure code, small synthetic tests, or compile checks. The human reviewer must be included before any step that runs training, downloads model weights, creates model checkpoints, or performs a longer GPU/CPU smoke run.

Required human checkpoints:

- Before running `python3 train_baseline.py`.
- Before any command that can download DINOv2 weights through `torch.hub`.
- Before changing debug subset values for a training smoke run.
- Before evaluating the reserved local test split.
- Before running any full training job beyond the tiny optional smoke run.

## File Structure

Create:

- `datasets/splits.py`: deterministic grouped split utilities.
- `engine/validation_records.py`: Kaggle-equivalent validation record building, instance extraction, direct scoring, and `recodai_f1.score` equivalence helpers.
- `tests/test_dataset_utils.py`: focused tests for sample discovery and mask helpers.
- `tests/test_splits.py`: focused tests for grouped split behavior.
- `tests/test_forgery_dataset.py`: focused tests for authentic zero-mask dataset behavior.
- `tests/test_validation_records.py`: focused tests for direct scorer semantics and official-score equivalence.

Modify:

- `dataset_utils.py`: add `SampleRecord`, labeled sample discovery, label-specific image lookup, path-based image loading, instance-mask loading, and union-mask-from-path helpers.
- `datasets/forgery_dataset.py`: accept `SampleRecord` objects, load images from exact paths, produce authentic all-zero masks.
- `engine/validate_loop.py`: return structured validation metrics and compute `kaggle_score`.
- `configs/baseline_config.py`: add split, metric, checkpoint, and post-processing config fields.
- `train_baseline.py`: replace forged-only split with grouped split, pass validation samples, save best checkpoint by `kaggle_score`.
- `inference/postprocess.py`: only extend configurability if the implementation needs the planned knobs; keep current defaults.
- `.gitignore`: add `/runs/` if checkpoints or split artifacts are written there.
- `main.py`: add a visible deprecation notice.
- `main.ipynb`: add a visible top markdown deprecation notice if notebook editing is practical.

Do not modify:

- `recodai-f1.py`: duplicate metric file with a hyphen in the filename.
- `max_individual_project/**`: reference only.
- `report/**`: draft report, not source of truth.

---

### Task 1: Add Dataset Utility Tests

**Files:**

- Create: `tests/test_dataset_utils.py`
- Modify: none
- Test: `tests/test_dataset_utils.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dataset_utils.py` with these tests:

```python
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
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
python3 -m unittest tests.test_dataset_utils -v
```

Expected: FAIL because `SampleRecord`, `list_labeled_samples`, `load_image_from_path`, `load_instance_masks`, and `load_union_mask_from_paths` do not exist yet.

- [ ] **Step 3: Commit checkpoint if commits are approved**

```bash
git add tests/test_dataset_utils.py
git commit -m "test: cover dataset sample discovery"
```

---

### Task 2: Implement Sample Metadata And Dataset Helpers

**Files:**

- Modify: `dataset_utils.py`
- Test: `tests/test_dataset_utils.py`

- [ ] **Step 1: Add `SampleRecord` and helper APIs**

Modify `dataset_utils.py` so it exposes these names and behaviors:

```python
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

DATA = Path("data")
LabelName = Literal["forged", "authentic"]
SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class SampleRecord:
    sample_id: str
    case_id: str
    label: LabelName
    image_path: Path
    mask_paths: tuple[Path, ...]
    group_id: str
    split: SplitName | None = None

    def with_split(self, split: SplitName) -> "SampleRecord":
        return replace(self, split=split)
```

Implementation requirements for the rest of the file:

- `find_image_path(case_id, label=None, data_root=DATA)` must search only the requested label directory when `label` is `"forged"` or `"authentic"`.
- When `label is None`, keep legacy behavior: forged first, authentic second, then `train_images/<case_id>.png`.
- `load_image_from_path(path)` must open the exact path, convert RGB to grayscale through `_to_gray`, and return `float32`.
- `find_mask_paths(case_id, data_root=DATA)` must keep deterministic mask sorting and support both `10.npy` and `10_*.npy`.
- `load_instance_masks(mask_paths_or_case_id)` must return a list of binary `uint8` masks.
- `load_union_mask_from_paths(mask_paths)` must OR all masks and return binary `uint8`.
- `load_union_mask(case_id)` should keep legacy behavior through `find_mask_paths`.
- `list_labeled_samples(data_root=DATA)` must return deterministic records sorted by `(sample_id)`, where authentic and forged records sharing a stem have unique IDs such as `authentic:10` and `forged:10`.
- Forged samples must fail with `FileNotFoundError` if no mask paths exist.
- Authentic samples must use an empty `mask_paths` tuple.

- [ ] **Step 2: Run the dataset utility tests**

Run:

```bash
python3 -m unittest tests.test_dataset_utils -v
```

Expected: PASS.

- [ ] **Step 3: Compile the module**

Run:

```bash
python3 -m py_compile dataset_utils.py
```

Expected: no output and exit code `0`.

- [ ] **Step 4: Commit checkpoint if commits are approved**

```bash
git add dataset_utils.py tests/test_dataset_utils.py
git commit -m "feat: add sample records and dataset helpers"
```

---

### Task 3: Add Grouped Split Tests

**Files:**

- Create: `tests/test_splits.py`
- Test: `tests/test_splits.py`

- [ ] **Step 1: Write failing grouped split tests**

Create `tests/test_splits.py`:

```python
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
```

- [ ] **Step 2: Run the split tests and verify they fail**

Run:

```bash
python3 -m unittest tests.test_splits -v
```

Expected: FAIL because `datasets/splits.py` does not exist yet.

- [ ] **Step 3: Commit checkpoint if commits are approved**

```bash
git add tests/test_splits.py
git commit -m "test: cover grouped stratified splits"
```

---

### Task 4: Implement Grouped Stratified Splits

**Files:**

- Create: `datasets/splits.py`
- Test: `tests/test_splits.py`

- [ ] **Step 1: Implement split utilities**

Create `datasets/splits.py` with these public functions:

```python
from __future__ import annotations

from collections import defaultdict
import random
from typing import Mapping, Sequence

from dataset_utils import SampleRecord

SplitDict = dict[str, list[SampleRecord]]


def group_samples_by_id(samples: Sequence[SampleRecord]) -> dict[str, list[SampleRecord]]:
    groups: dict[str, list[SampleRecord]] = defaultdict(list)
    for sample in samples:
        groups[sample.group_id].append(sample)
    return {group_id: sorted(group_samples, key=lambda item: item.sample_id) for group_id, group_samples in groups.items()}
```

The rest of `datasets/splits.py` must implement:

- `make_grouped_stratified_splits(samples, seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1) -> SplitDict`
- `count_samples_by_split_and_label(splits: Mapping[str, Sequence[SampleRecord]]) -> dict`

Implementation rules:

- Validate that ratios sum to `1.0` within floating-point tolerance.
- Assign each group a type: `paired`, `forged_only`, or `authentic_only`.
- Sort group IDs before shuffling so the seed is reproducible.
- Shuffle groups within each type with `random.Random(seed)`.
- Allocate each type independently to train/val/test using rounded counts that preserve all groups.
- Expand groups back into records and set `sample.split` with `sample.with_split(split_name)`.
- Sort samples inside each split by `(group_id, label, sample_id)` after expansion.

- [ ] **Step 2: Run split tests**

Run:

```bash
python3 -m unittest tests.test_splits -v
```

Expected: PASS.

- [ ] **Step 3: Compile split module**

Run:

```bash
python3 -m py_compile datasets/splits.py
```

Expected: no output and exit code `0`.

- [ ] **Step 4: Commit checkpoint if commits are approved**

```bash
git add datasets/splits.py tests/test_splits.py
git commit -m "feat: add grouped stratified splits"
```

---

### Task 5: Add Dataset Authentic Support Tests

**Files:**

- Create: `tests/test_forgery_dataset.py`
- Test: `tests/test_forgery_dataset.py`

- [ ] **Step 1: Write failing dataset behavior tests**

Create `tests/test_forgery_dataset.py`:

```python
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
```

- [ ] **Step 2: Run the dataset tests and verify they fail**

Run:

```bash
python3 -m unittest tests.test_forgery_dataset -v
```

Expected: FAIL because `ForgeryDataset` still expects case IDs and calls `load_image(case_id)` plus `load_union_mask(case_id)`.

- [ ] **Step 3: Commit checkpoint if commits are approved**

```bash
git add tests/test_forgery_dataset.py
git commit -m "test: cover authentic dataset masks"
```

---

### Task 6: Refactor `ForgeryDataset`

**Files:**

- Modify: `datasets/forgery_dataset.py`
- Test: `tests/test_forgery_dataset.py`, `tests/test_dataset_utils.py`

- [ ] **Step 1: Update dataset loading**

Modify `datasets/forgery_dataset.py`:

- constructor accepts `samples` instead of `case_ids`,
- store `self.samples = list(samples)`,
- keep resize, RGB replication, normalization, tensor shape, and default return `(img, mask)`,
- for each sample, call `load_image_from_path(sample.image_path)`,
- for forged samples, call `load_union_mask_from_paths(sample.mask_paths)`,
- for authentic samples, create `np.zeros(img.shape[:2], dtype=np.uint8)`,
- do not use `case_id` lookup when a `SampleRecord` is available.

The import section should use:

```python
from dataset_utils import SampleRecord, load_image_from_path, load_union_mask_from_paths
```

- [ ] **Step 2: Run dataset tests**

Run:

```bash
python3 -m unittest tests.test_forgery_dataset tests.test_dataset_utils -v
```

Expected: PASS.

- [ ] **Step 3: Compile dataset module**

Run:

```bash
python3 -m py_compile datasets/forgery_dataset.py
```

Expected: no output and exit code `0`.

- [ ] **Step 4: Commit checkpoint if commits are approved**

```bash
git add datasets/forgery_dataset.py tests/test_forgery_dataset.py
git commit -m "feat: support authentic samples in dataset"
```

---

### Task 7: Add Validation Scoring Tests

**Files:**

- Create: `tests/test_validation_records.py`
- Test: `tests/test_validation_records.py`

- [ ] **Step 1: Write failing validation tests**

Create `tests/test_validation_records.py`:

```python
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
```

- [ ] **Step 2: Run validation tests and verify they fail**

Run:

```bash
python3 -m unittest tests.test_validation_records -v
```

Expected: FAIL because `engine/validation_records.py` does not exist yet.

- [ ] **Step 3: Commit checkpoint if commits are approved**

```bash
git add tests/test_validation_records.py
git commit -m "test: cover kaggle-equivalent validation scoring"
```

---

### Task 8: Implement Validation Record And Scoring Helpers

**Files:**

- Create: `engine/validation_records.py`
- Test: `tests/test_validation_records.py`

- [ ] **Step 1: Implement validation helper APIs**

Create `engine/validation_records.py` with these public functions:

```python
from __future__ import annotations

import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from dataset_utils import SampleRecord, load_instance_masks
from recodai_f1 import oF1_score, rle_encode, score
```

Required behavior:

- `resize_binary_mask(mask, shape)` resizes one binary mask to `(height, width)` with nearest-neighbor interpolation and returns `uint8`.
- `load_resized_instance_masks(sample, shape)` returns `[]` for authentic samples and resized individual masks for forged samples.
- `connected_components_to_masks(mask)` returns one binary `uint8` mask per connected component, sorted by component ID.
- `mask_instances_to_annotation(instances)` returns `"authentic"` for an empty list and `rle_encode(instances)` otherwise.
- `score_instances(pred_instances, gt_instances)` implements official image-level semantics around `oF1_score`.
- `compute_kaggle_score_from_instances(ordered_samples, pred_instances_by_sample_id, gt_instances_by_sample_id)` averages `score_instances` in the exact provided order.
- `build_solution_rows(ordered_samples, gt_instances_by_sample_id, shapes_by_sample_id)` returns a pandas DataFrame with `sample_id`, `annotation`, and `shape`.
- `build_submission_rows(ordered_samples, pred_instances_by_sample_id)` returns a pandas DataFrame with `sample_id` and `annotation`.
- `compute_kaggle_score_via_recodai(solution, submission)` calls `score(solution.copy(), submission.copy(), row_id_column_name="sample_id")`.

Official image-level semantics:

```python
def score_instances(pred_instances: Sequence[np.ndarray], gt_instances: Sequence[np.ndarray]) -> float:
    pred_empty = len(pred_instances) == 0
    gt_empty = len(gt_instances) == 0
    if gt_empty and pred_empty:
        return 1.0
    if gt_empty and not pred_empty:
        return 0.0
    if not gt_empty and pred_empty:
        return 0.0
    return float(oF1_score(list(pred_instances), list(gt_instances)))
```

- [ ] **Step 2: Run validation tests**

Run:

```bash
python3 -m unittest tests.test_validation_records -v
```

Expected: PASS.

- [ ] **Step 3: Compile validation helper module**

Run:

```bash
python3 -m py_compile engine/validation_records.py
```

Expected: no output and exit code `0`.

- [ ] **Step 4: Commit checkpoint if commits are approved**

```bash
git add engine/validation_records.py tests/test_validation_records.py
git commit -m "feat: add kaggle-equivalent validation helpers"
```

---

### Task 9: Refactor Validation Loop

**Files:**

- Modify: `engine/validate_loop.py`
- Test: `tests/test_validation_records.py`

- [ ] **Step 1: Change validation signature and return type**

Update `validate_one_epoch` so the signature accepts:

```python
def validate_one_epoch(
    model: torch.nn.Module,
    val_loader,
    val_samples,
    device: torch.device,
    sliding_window_fn,
    pixel_util,
    pred_threshold: float,
    harden_temperature: float,
    hard_clip_low: float,
    hard_clip_high: float,
    min_component_area: int,
    epoch_idx: int,
    compute_pixel_f1: bool = False,
    verify_score_equivalence: bool = False,
) -> dict:
```

Required behavior:

- Require `val_loader` order to match `val_samples`; training code must use `shuffle=False`.
- Track a running `sample_index` while iterating through batches.
- For each prediction, run the current sigmoid and `post_process_prediction` path.
- Convert `pred_bin = (probs >= pred_threshold).astype(np.uint8)` to prediction instances with `connected_components_to_masks`.
- Build ground-truth instances with `load_resized_instance_masks(sample, pred_bin.shape)`.
- Accumulate `pred_instances_by_sample_id`, `gt_instances_by_sample_id`, and `shapes_by_sample_id`.
- Compute `kaggle_score` by calling `compute_kaggle_score_from_instances` with `ordered_samples=val_samples`, `pred_instances_by_sample_id=pred_instances_by_sample_id`, and `gt_instances_by_sample_id=gt_instances_by_sample_id`.
- If `verify_score_equivalence` is true, build DataFrames and compare direct score with `compute_kaggle_score_via_recodai`; raise `AssertionError` when the absolute difference is greater than `1e-8`.
- If `compute_pixel_f1` is true, compute the old union-mask pixel F1 and return it as `pixel_f1`.
- Return:

```python
{
    "kaggle_score": float(kaggle_score),
    "pixel_f1": pixel_f1_or_none,
    "num_samples": len(val_samples),
    "num_forged": sum(1 for sample in val_samples if sample.label == "forged"),
    "num_authentic": sum(1 for sample in val_samples if sample.label == "authentic"),
}
```

- [ ] **Step 2: Compile validation loop**

Run:

```bash
python3 -m py_compile engine/validate_loop.py
```

Expected: no output and exit code `0`.

- [ ] **Step 3: Run validation helper tests**

Run:

```bash
python3 -m unittest tests.test_validation_records -v
```

Expected: PASS.

- [ ] **Step 4: Commit checkpoint if commits are approved**

```bash
git add engine/validate_loop.py
git commit -m "feat: validate with kaggle score"
```

---

### Task 10: Add Config Fields

**Files:**

- Modify: `configs/baseline_config.py`
- Test: compile check

- [ ] **Step 1: Add split, metric, and checkpoint config fields**

Add these fields to `BaselineConfig` while keeping existing model/training defaults:

```python
data_root: str = "data"
train_ratio: float = 0.8
val_ratio: float = 0.1
test_ratio: float = 0.1
train_subset: int | None = None
val_subset: int | None = None
compute_pixel_f1: bool = False
verify_score_equivalence: bool = False
checkpoint_dir: str = "runs/checkpoints"
best_checkpoint_name: str = "best_by_kaggle_score.pt"
include_supplemental: bool = False
post_process_confident_threshold: float | None = None
post_process_smooth_probabilities: bool = False
post_process_fill_holes: bool = True
post_process_apply_opening: bool = True
post_process_apply_closing: bool = False
post_process_keep_confident_seeded_components: bool = False
```

Keep the existing fields:

```python
pred_threshold: float = 0.5
harden_temperature: float = 0.7
hard_clip_low: float = 0.1
hard_clip_high: float = 0.9
min_component_area: int = 50
```

- [ ] **Step 2: Compile config**

Run:

```bash
python3 -m py_compile configs/baseline_config.py
```

Expected: no output and exit code `0`.

- [ ] **Step 3: Commit checkpoint if commits are approved**

```bash
git add configs/baseline_config.py
git commit -m "feat: add baseline refactor config fields"
```

---

### Task 11: Refactor Training Orchestration

**Files:**

- Modify: `train_baseline.py`
- Modify: `.gitignore` if `/runs/` is missing
- Test: compile check and dry data inventory command

- [ ] **Step 1: Replace forged-only discovery with sample discovery and splits**

Modify `train_baseline.py` imports:

```python
from dataset_utils import list_labeled_samples
from datasets.splits import count_samples_by_split_and_label, make_grouped_stratified_splits
```

Remove normal use of `get_forged_case_ids()` and `split_ids()`. In `main()`, create splits:

```python
all_samples = list_labeled_samples(Path(config.data_root))
splits = make_grouped_stratified_splits(
    all_samples,
    seed=config.seed,
    train_ratio=config.train_ratio,
    val_ratio=config.val_ratio,
    test_ratio=config.test_ratio,
)
train_samples = splits["train"]
val_samples = splits["val"]
test_samples = splits["test"]
```

Apply debug subsets only after splitting:

```python
if config.train_subset is not None:
    print(f"Debug train_subset enabled: using {config.train_subset} of {len(train_samples)} train samples")
    train_samples = train_samples[: config.train_subset]
if config.val_subset is not None:
    print(f"Debug val_subset enabled: using {config.val_subset} of {len(val_samples)} val samples")
    val_samples = val_samples[: config.val_subset]
```

Use `ForgeryDataset(train_samples, config.target_size, use_rgb=config.use_rgb, normalize_rgb=config.normalize_rgb, rgb_mean=config.dino_mean, rgb_std=config.dino_std)` and `ForgeryDataset(val_samples, config.target_size, use_rgb=config.use_rgb, normalize_rgb=config.normalize_rgb, rgb_mean=config.dino_mean, rgb_std=config.dino_std)`.

- [ ] **Step 2: Wire validation result into scheduler and checkpointing**

Change validation call:

```python
validation_result = validate_one_epoch(
    model=model,
    val_loader=val_loader,
    val_samples=val_samples,
    device=device,
    sliding_window_fn=sliding_window_fn,
    pixel_util=pixel_util,
    pred_threshold=config.pred_threshold,
    harden_temperature=config.harden_temperature,
    hard_clip_low=config.hard_clip_low,
    hard_clip_high=config.hard_clip_high,
    min_component_area=config.min_component_area,
    epoch_idx=epoch,
    compute_pixel_f1=config.compute_pixel_f1,
    verify_score_equivalence=config.verify_score_equivalence,
)
kaggle_score = validation_result["kaggle_score"]
```

Rename `best_f1` to `best_kaggle_score`. Scheduler uses `kaggle_score`.

Save best checkpoint:

```python
checkpoint_dir = Path(config.checkpoint_dir)
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = checkpoint_dir / config.best_checkpoint_name
torch.save(
    {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "kaggle_score": kaggle_score,
        "validation_result": validation_result,
        "config": config.__dict__,
        "split_counts": split_counts,
        "model_name": config.dino_model_name,
    },
    checkpoint_path,
)
```

- [ ] **Step 3: Ensure generated run artifacts are ignored**

If `.gitignore` does not contain `/runs/`, add:

```text
/runs/
```

- [ ] **Step 4: Compile training script**

Run:

```bash
python3 -m py_compile train_baseline.py
```

Expected: no output and exit code `0`.

- [ ] **Step 5: Run a no-training inventory check**

Run:

```bash
python3 -c "from pathlib import Path; from dataset_utils import list_labeled_samples; from datasets.splits import make_grouped_stratified_splits, count_samples_by_split_and_label; samples=list_labeled_samples(Path('data')); splits=make_grouped_stratified_splits(samples, seed=42); print(count_samples_by_split_and_label(splits))"
```

Expected current local totals include `5128` total samples, `2751` forged, `2377` authentic, and train/val/test split counts that sum to `5128`.

- [ ] **Step 6: Commit checkpoint if commits are approved**

```bash
git add train_baseline.py .gitignore
git commit -m "feat: train baseline with grouped splits"
```

---

### Task 12: Run Integrated Smoke Checks With Human Approval For Training

**Files:**

- Modify: none unless a smoke check reveals an implementation defect
- Test: all focused tests and compile checks

- [ ] **Step 1: Run all unit-style tests**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: PASS.

- [ ] **Step 2: Run compile checks for changed modules**

Run:

```bash
python3 -m py_compile dataset_utils.py datasets/forgery_dataset.py datasets/splits.py engine/validation_records.py engine/validate_loop.py configs/baseline_config.py train_baseline.py
```

Expected: no output and exit code `0`.

- [ ] **Step 3: Human approval checkpoint before any training smoke run**

Ask the human reviewer before continuing:

```text
The compile checks and unit-style tests passed. Do you want me to run the optional tiny training smoke check now?

This may initialize DINOv2 through torch.hub, download weights if they are not already cached, use CPU/GPU time, and write a checkpoint under runs/checkpoints/.
```

Expected: continue only if the human explicitly approves the smoke run.

- [ ] **Step 4: Prepare one debug training epoch only after approval**

Temporarily set these config values in `configs/baseline_config.py` for the smoke run, then restore them in the same implementation step:

```python
num_epochs: int = 1
train_subset: int | None = 8
val_subset: int | None = 4
verify_score_equivalence: bool = True
compute_pixel_f1: bool = True
```

- [ ] **Step 5: Run the approved debug training smoke check**

Run:

```bash
python3 train_baseline.py
```

Expected:

- training starts,
- validation logs `kaggle_score`,
- `pixel_f1` appears only because `compute_pixel_f1` is true,
- best checkpoint is written under `runs/checkpoints/best_by_kaggle_score.pt`,
- no local test split is used for model selection.

- [ ] **Step 6: Restore normal config after the smoke run**

Restore:

```python
num_epochs: int = 10
train_subset: int | None = None
val_subset: int | None = None
verify_score_equivalence: bool = False
compute_pixel_f1: bool = False
```

- [ ] **Step 7: Commit checkpoint if commits are approved**

```bash
git add dataset_utils.py datasets/forgery_dataset.py datasets/splits.py engine/validation_records.py engine/validate_loop.py configs/baseline_config.py train_baseline.py tests
git commit -m "test: verify baseline refactor smoke path"
```

---

### Task 13: Mark Deprecated Entrypoints

**Files:**

- Modify: `main.py`
- Modify: `main.ipynb` if safe
- Test: visual/readability check

- [ ] **Step 1: Mark `main.py` deprecated**

Add a top-level module docstring at the start of `main.py`:

```python
"""
Deprecated exploratory entrypoint.

The maintained baseline training entrypoint is train_baseline.py.
This file is kept only for historical context and should not be used
for current experiments.
"""
```

- [ ] **Step 2: Mark `main.ipynb` deprecated if notebook editing is practical**

Add a top markdown cell:

```markdown
# Deprecated exploratory notebook

The maintained baseline training entrypoint is `train_baseline.py`.
This notebook is kept for historical context and should not be used as the current baseline implementation.
```

- [ ] **Step 3: Commit checkpoint if commits are approved**

```bash
git add main.py main.ipynb
git commit -m "docs: mark deprecated baseline prototypes"
```

---

## Final Review Checklist

- [ ] `train_baseline.py` uses `list_labeled_samples` and grouped splits.
- [ ] Normal baseline runs use all split samples because `train_subset` and `val_subset` default to `None`.
- [ ] Authentic records train with all-zero masks.
- [ ] Forged records preserve individual mask paths for validation.
- [ ] Matching authentic/forged stems stay in the same split.
- [ ] Validation returns a dictionary with `kaggle_score`.
- [ ] Best checkpoint selection uses `kaggle_score`.
- [ ] `pixel_f1` is opt-in and labeled as non-official when printed.
- [ ] Direct validation score is verified against `recodai_f1.score` by smoke tests.
- [ ] Local test split is not used in the training loop.
- [ ] `main.py` and `main.ipynb` are clearly deprecated.
- [ ] Generated artifacts under `runs/` are ignored.
- [ ] `recodai-f1.py` is not imported by new code.

## Self-Review

Spec coverage:

- Workstream 1 is covered by Tasks 7, 8, 9, 10, 11, and 12.
- Workstream 2 is covered by Tasks 1 through 6 and Task 11.
- Workstream 3 is covered by Task 13.
- Workstream 4 is intentionally not implemented here; its decisions are respected by keeping `torch.hub` training and avoiding Hugging Face/TTA/global-local changes in this chunk.
- Workstreams 5 and 6 are intentionally deferred until code behavior is stable.
- Workstream 7 is intentionally excluded because performance experiments should not be mixed with metric/data correctness.

Placeholder scan:

- This note avoids open placeholders and names exact files, functions, commands, and expected outcomes.
- Out-of-scope items are named as exclusions, not as unfinished implementation steps.

Type consistency:

- `SampleRecord.sample_id`, `case_id`, `label`, `image_path`, `mask_paths`, `group_id`, and `split` are used consistently across dataset, split, validation, and training tasks.
- The validation result uses `kaggle_score`, `pixel_f1`, `num_samples`, `num_forged`, and `num_authentic` consistently.
- Split names are consistently `train`, `val`, and `test`.

## Execution Handoff

Plan complete and saved to `docs/implementation_note.md`.

Two execution options when the user gives the direct implementation signal:

1. Subagent-driven execution: dispatch a fresh worker per task, review each task before moving on.
2. Inline execution: implement the tasks in this session with checkpoints after each task group.

No implementation should begin until the user explicitly chooses to start.
