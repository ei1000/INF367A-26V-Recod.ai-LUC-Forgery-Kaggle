# Validation Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make epoch validation faster and smoother by separating GPU inference, CPU post-processing, and Kaggle-style scoring while keeping full `kaggle_score` validation every epoch.

**Architecture:** Validation becomes a three-phase pipeline. Phase 1 performs batched GPU inference and stores CPU probability maps. Phase 2 post-processes those probability maps, extracts predicted instances, and loads/resizes ground-truth instances on CPU. Phase 3 computes the existing direct Kaggle-equivalent score and optional official parity check.

**Tech Stack:** Python 3.12, PyTorch, NumPy, SciPy, PIL, existing `recodai_f1.py`, existing `PixelMapUtil`, `unittest`.

---

## Scope

This plan covers epoch-validation performance and structure only. It keeps validation every epoch and keeps model selection based on `kaggle_score`.

This plan does not switch DINOv2 Base to DINOv2 Small. That change touches model configuration, embedding dimensions, checkpoint compatibility, and performance comparisons, so it should get its own short plan after validation is stable.

This plan does not add the Colab notebook. Colab setup is independent from validation internals and should follow as `docs/colab_training_plan.md` or a direct `notebooks/colab_train_baseline.ipynb` plan after this refactor.

## Research Notes

Current repository behavior:

- `engine/validate_loop.py` loops over a validation batch, then validates one image at a time.
- The current loop calls `sliding_window_fn(img, model, device)` for each image.
- The current loop immediately calls `.cpu().numpy()` after each image, then runs CPU post-processing and instance work before the next image.
- `configs/baseline_config.py` currently sets `target_size = 448` and `sliding_window_size = 448`.
- `inference/sliding_window_dino_impl.py` currently uses `range(0, h_img, stride)` and `range(0, w_img, stride)`, so a 448x448 image with stride 224 produces four crops instead of one.

Inspiration notebook behavior:

- It uses a small-image fast path: if the image fits inside the window, it pads once and runs one prediction.
- It batches crops for inference.
- It builds a full probability map before CPU post-processing.
- It is an inference notebook, not a training pipeline, so only the batching and fast-path structure should be reused.

## File Structure

- Create `tests/test_sliding_window_dino.py`: tests crop-start generation and exact-size fast path using a tiny fake model.
- Create `tests/test_validation_inference.py`: tests batched GPU inference helpers with fake models and synthetic loader batches.
- Create `engine/validation_inference.py`: owns validation GPU inference collection and the CPU-side prediction dataclass.
- Modify `inference/sliding_window_dino_impl.py`: add deterministic window-start helper, exact-size fast path, cached Gaussian weights, and duplicate-coordinate protection.
- Modify `engine/validate_loop.py`: reduce it to orchestration over GPU inference, CPU post-processing, and scoring.
- Modify `configs/baseline_config.py`: add validation inference mode, probability dtype, and timing-log fields.
- Modify `train_baseline.py`: pass new config fields into `validate_one_epoch`.

## Behavioral Contract

- Default epoch validation uses direct batched inference with `model(imgs)`.
- Optional sliding-window validation remains available for high-resolution validation experiments.
- Validation still runs every epoch.
- Best checkpoint selection still uses `validation_result["kaggle_score"]`.
- `compute_pixel_f1` remains optional and clearly non-official.
- `verify_score_equivalence` remains optional and compares direct score against `recodai_f1.score`.
- No training command is part of this plan. Human approval is required before running `python train_baseline.py`.

---

### Task 1: Sliding-Window Crop Starts And Fast Path Tests

**Files:**
- Create: `tests/test_sliding_window_dino.py`
- Modify: none
- Test: `tests/test_sliding_window_dino.py`

- [ ] **Step 1: Write failing tests for crop starts and exact-size inference**

Create `tests/test_sliding_window_dino.py` with this content:

```python
import unittest

import torch

from inference.sliding_window_dino_impl import compute_window_starts, sliding_window_dino


class CountingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.batch_sizes: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.batch_sizes.append(int(x.shape[0]))
        return torch.ones((x.shape[0], 1, x.shape[-2], x.shape[-1]), device=x.device)


class SlidingWindowDinoTests(unittest.TestCase):
    def test_compute_window_starts_exact_size_uses_single_start(self) -> None:
        self.assertEqual(compute_window_starts(length=448, patch_size=448, stride=224), [0])

    def test_compute_window_starts_smaller_than_patch_uses_single_start(self) -> None:
        self.assertEqual(compute_window_starts(length=320, patch_size=448, stride=224), [0])

    def test_compute_window_starts_larger_image_includes_final_aligned_start_without_duplicates(self) -> None:
        starts = compute_window_starts(length=1000, patch_size=448, stride=224)

        self.assertEqual(starts, [0, 224, 448, 552])
        self.assertEqual(len(starts), len(set(starts)))
        self.assertEqual(starts[-1] + 448, 1000)

    def test_sliding_window_exact_size_calls_model_once(self) -> None:
        model = CountingModel()
        img = torch.zeros((3, 448, 448))

        pred = sliding_window_dino(
            img=img,
            model=model,
            device=torch.device("cpu"),
            patch_size=448,
            stride=224,
            batch_size=8,
        )

        self.assertEqual(tuple(pred.shape), (448, 448))
        self.assertEqual(model.calls, 1)
        self.assertEqual(model.batch_sizes, [1])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_sliding_window_dino -v
```

Expected:

```text
ImportError: cannot import name 'compute_window_starts'
```

- [ ] **Step 3: Commit checkpoint if commits are approved**

```bash
git add tests/test_sliding_window_dino.py
git commit -m "test: cover validation sliding-window crop starts"
```

---

### Task 2: Sliding-Window Start Helper And Exact-Size Fast Path

**Files:**
- Modify: `inference/sliding_window_dino_impl.py`
- Test: `tests/test_sliding_window_dino.py`

- [ ] **Step 1: Add deterministic crop starts and cached Gaussian weights**

Replace the contents of `inference/sliding_window_dino_impl.py` with:

```python
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

PATCH_SIZE = 256
BATCH_SIZE = 32
EPS = 1e-5
STRIDE = PATCH_SIZE // 2


def compute_window_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if length <= patch_size:
        return [0]

    starts = list(range(0, max(1, length - patch_size + 1), stride))
    final_start = length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return sorted(set(starts))


@lru_cache(maxsize=16)
def gaussian_weight_numpy(patch_size: int, sigma: float = 0.125) -> np.ndarray:
    ax = np.linspace(-1, 1, patch_size)
    xx, yy = np.meshgrid(ax, ax)
    dist = np.sqrt(xx**2 + yy**2)
    return np.exp(-(dist**2) / (2 * sigma**2)).astype(np.float32)


def gaussian_weight(patch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.from_numpy(gaussian_weight_numpy(patch_size)).to(device=device, dtype=dtype)


def predict_batched_crops(crops: list[torch.Tensor], model, device: torch.device) -> torch.Tensor:
    batch = torch.stack(crops, dim=0).to(device, non_blocking=True)
    return model(batch)


def _pad_crop_to_patch(crop: torch.Tensor, patch_size: int) -> torch.Tensor:
    pad_h = max(0, patch_size - crop.shape[1])
    pad_w = max(0, patch_size - crop.shape[2])
    if pad_h == 0 and pad_w == 0:
        return crop
    return F.pad(crop, (0, pad_w, 0, pad_h), mode="constant")


def sliding_window_dino(
    img,
    model,
    device,
    patch_size: int = PATCH_SIZE,
    stride: int | None = None,
    batch_size: int = BATCH_SIZE,
):
    if img.ndim != 3:
        raise ValueError(f"Expected image shape (C,H,W), got {tuple(img.shape)}")

    if stride is None:
        stride = patch_size // 2

    h_img, w_img = int(img.shape[-2]), int(img.shape[-1])

    if h_img <= patch_size and w_img <= patch_size:
        patch = _pad_crop_to_patch(img, patch_size)
        pred = model(patch[None].to(device, non_blocking=True))[0].squeeze(0)
        return pred[:h_img, :w_img]

    y_starts = compute_window_starts(h_img, patch_size, stride)
    x_starts = compute_window_starts(w_img, patch_size, stride)

    weight = gaussian_weight(patch_size, device=device, dtype=torch.float32)
    prob_map = torch.zeros((h_img, w_img), device=device, dtype=torch.float32)
    weight_map = torch.zeros((h_img, w_img), device=device, dtype=torch.float32) + EPS

    crops: list[torch.Tensor] = []
    coords: list[tuple[int, int]] = []
    for y in y_starts:
        for x in x_starts:
            crop = img[:, y : y + patch_size, x : x + patch_size]
            crops.append(_pad_crop_to_patch(crop, patch_size))
            coords.append((y, x))

    model.eval()
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i : i + batch_size]
            batch_coords = coords[i : i + batch_size]
            preds = predict_batched_crops(batch_crops, model, device)

            for pred, (y, x) in zip(preds, batch_coords):
                pred = pred.squeeze(0)
                h = min(patch_size, h_img - y)
                w = min(patch_size, w_img - x)
                prob_map[y : y + h, x : x + w] += pred[:h, :w] * weight[:h, :w]
                weight_map[y : y + h, x : x + w] += weight[:h, :w]

    return prob_map / weight_map
```

- [ ] **Step 2: Run the focused sliding-window tests**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_sliding_window_dino -v
```

Expected:

```text
Ran 4 tests
OK
```

- [ ] **Step 3: Run compile check**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache PYTHONPYCACHEPREFIX=/tmp/pycache uv run python -m py_compile inference/sliding_window_dino_impl.py
```

Expected: command exits with status 0.

- [ ] **Step 4: Commit checkpoint if commits are approved**

```bash
git add inference/sliding_window_dino_impl.py tests/test_sliding_window_dino.py
git commit -m "fix: avoid duplicate validation sliding-window crops"
```

---

### Task 3: Batched Validation Inference Tests

**Files:**
- Create: `tests/test_validation_inference.py`
- Modify: none
- Test: `tests/test_validation_inference.py`

- [ ] **Step 1: Write failing tests for direct batched validation inference**

Create `tests/test_validation_inference.py` with this content:

```python
from pathlib import Path
import unittest

import numpy as np
import torch

from dataset_utils import SampleRecord
from engine.validation_inference import ValidationPrediction, collect_validation_predictions


def _sample(sample_id: str, label: str = "authentic") -> SampleRecord:
    return SampleRecord(
        sample_id=f"{label}:{sample_id}",
        case_id=sample_id,
        label=label,
        image_path=Path(f"data/train_images/{label}/{sample_id}.png"),
        mask_paths=tuple(),
        group_id=sample_id,
        split="val",
    )


class BatchCountingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(x.shape[0]))
        logits = torch.zeros((x.shape[0], 1, x.shape[-2], x.shape[-1]), device=x.device)
        logits[:, :, 1:3, 1:3] = 4.0
        return logits


class ValidationInferenceTests(unittest.TestCase):
    def test_collect_direct_predictions_runs_one_forward_per_loader_batch(self) -> None:
        model = BatchCountingModel()
        samples = [_sample("1"), _sample("2"), _sample("3")]
        imgs_1 = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
        masks_1 = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
        imgs_2 = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        masks_2 = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        loader = [(imgs_1, masks_1), (imgs_2, masks_2)]

        predictions = collect_validation_predictions(
            model=model,
            val_loader=loader,
            val_samples=samples,
            device=torch.device("cpu"),
            inference_mode="direct",
            sliding_window_fn=None,
            probability_dtype="float32",
            collect_masks=False,
        )

        self.assertEqual(model.batch_sizes, [2, 1])
        self.assertEqual([p.sample.sample_id for p in predictions], ["authentic:1", "authentic:2", "authentic:3"])
        self.assertTrue(all(isinstance(p, ValidationPrediction) for p in predictions))
        self.assertTrue(all(p.probability.shape == (8, 8) for p in predictions))
        self.assertTrue(all(p.gt_union_mask is None for p in predictions))

    def test_collect_direct_predictions_can_keep_cpu_masks_for_pixel_f1(self) -> None:
        model = BatchCountingModel()
        samples = [_sample("1"), _sample("2")]
        imgs = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
        masks = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
        masks[1, 0, 4:6, 4:6] = 1.0
        loader = [(imgs, masks)]

        predictions = collect_validation_predictions(
            model=model,
            val_loader=loader,
            val_samples=samples,
            device=torch.device("cpu"),
            inference_mode="direct",
            sliding_window_fn=None,
            probability_dtype="float16",
            collect_masks=True,
        )

        self.assertEqual(predictions[0].probability.dtype, np.float16)
        self.assertIsNotNone(predictions[0].gt_union_mask)
        self.assertEqual(int(predictions[1].gt_union_mask.sum()), 4)

    def test_collect_predictions_rejects_missing_sliding_window_function_in_sliding_mode(self) -> None:
        model = BatchCountingModel()
        samples = [_sample("1")]
        imgs = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        masks = torch.zeros((1, 1, 8, 8), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "sliding_window_fn"):
            collect_validation_predictions(
                model=model,
                val_loader=[(imgs, masks)],
                val_samples=samples,
                device=torch.device("cpu"),
                inference_mode="sliding",
                sliding_window_fn=None,
                probability_dtype="float32",
                collect_masks=False,
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_validation_inference -v
```

Expected:

```text
ModuleNotFoundError: No module named 'engine.validation_inference'
```

- [ ] **Step 3: Commit checkpoint if commits are approved**

```bash
git add tests/test_validation_inference.py
git commit -m "test: cover batched validation inference collection"
```

---

### Task 4: Batched Validation Inference Helper

**Files:**
- Create: `engine/validation_inference.py`
- Test: `tests/test_validation_inference.py`

- [ ] **Step 1: Create validation inference helper**

Create `engine/validation_inference.py` with this content:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

from dataset_utils import SampleRecord


@dataclass(frozen=True, slots=True)
class ValidationPrediction:
    sample: SampleRecord
    probability: np.ndarray
    gt_union_mask: np.ndarray | None = None


def _numpy_dtype(name: str) -> np.dtype:
    if name == "float16":
        return np.dtype(np.float16)
    if name == "float32":
        return np.dtype(np.float32)
    raise ValueError("probability_dtype must be 'float16' or 'float32'")


def _squeeze_probability_batch(probs: torch.Tensor) -> torch.Tensor:
    if probs.ndim == 4 and probs.shape[1] == 1:
        return probs[:, 0]
    if probs.ndim == 3:
        return probs
    raise ValueError(f"Expected probability tensor shape (B,1,H,W) or (B,H,W), got {tuple(probs.shape)}")


@torch.inference_mode()
def collect_validation_predictions(
    model: torch.nn.Module,
    val_loader,
    val_samples: Sequence[SampleRecord],
    device: torch.device,
    inference_mode: str,
    sliding_window_fn: Callable | None,
    probability_dtype: str = "float16",
    collect_masks: bool = False,
) -> list[ValidationPrediction]:
    dtype = _numpy_dtype(probability_dtype)
    predictions: list[ValidationPrediction] = []
    sample_index = 0

    model.eval()

    for imgs, masks in val_loader:
        batch_size = int(imgs.shape[0])
        batch_samples = list(val_samples[sample_index : sample_index + batch_size])
        if len(batch_samples) != batch_size:
            raise ValueError(
                f"val_loader yielded more samples than val_samples: index={sample_index}, batch_size={batch_size}"
            )
        sample_index += batch_size

        if inference_mode == "direct":
            imgs_device = imgs.to(device, non_blocking=True)
            logits = model(imgs_device)
            probs_tensor = torch.sigmoid(logits)
            probs_np = _squeeze_probability_batch(probs_tensor).detach().cpu().numpy().astype(dtype, copy=False)
        elif inference_mode == "sliding":
            if sliding_window_fn is None:
                raise ValueError("sliding_window_fn is required when inference_mode='sliding'")
            imgs_device = imgs.to(device, non_blocking=True)
            per_image_probs = []
            for img in imgs_device:
                logits = sliding_window_fn(img, model, device)
                per_image_probs.append(torch.sigmoid(logits).squeeze(0))
            probs_np = torch.stack(per_image_probs, dim=0).detach().cpu().numpy().astype(dtype, copy=False)
        else:
            raise ValueError("inference_mode must be 'direct' or 'sliding'")

        masks_np = None
        if collect_masks:
            masks_np = masks.detach().cpu().numpy()[:, 0].astype(np.uint8, copy=False)

        for batch_offset, sample in enumerate(batch_samples):
            gt_union_mask = masks_np[batch_offset] if masks_np is not None else None
            predictions.append(
                ValidationPrediction(
                    sample=sample,
                    probability=probs_np[batch_offset],
                    gt_union_mask=gt_union_mask,
                )
            )

    if sample_index != len(val_samples):
        raise ValueError(f"val_loader yielded {sample_index} samples but val_samples has {len(val_samples)}")

    return predictions
```

- [ ] **Step 2: Run validation inference tests**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_validation_inference -v
```

Expected:

```text
Ran 3 tests
OK
```

- [ ] **Step 3: Run compile check**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache PYTHONPYCACHEPREFIX=/tmp/pycache uv run python -m py_compile engine/validation_inference.py
```

Expected: command exits with status 0.

- [ ] **Step 4: Commit checkpoint if commits are approved**

```bash
git add engine/validation_inference.py tests/test_validation_inference.py
git commit -m "feat: collect validation probabilities in GPU batches"
```

---

### Task 5: CPU Scoring Phase Tests

**Files:**
- Modify: `tests/test_validation_inference.py`
- Test: `tests/test_validation_inference.py`

- [ ] **Step 1: Add CPU scoring tests**

Append these imports to the existing import block in `tests/test_validation_inference.py`:

```python
from unittest.mock import patch

from engine.validation_inference import score_validation_predictions
```

Append these tests inside `ValidationInferenceTests`:

```python
    def test_score_validation_predictions_postprocesses_after_prediction_collection(self) -> None:
        authentic = _sample("1", label="authentic")
        forged = SampleRecord(
            sample_id="forged:2",
            case_id="2",
            label="forged",
            image_path=Path("data/train_images/forged/2.png"),
            mask_paths=(Path("data/train_masks/2.npy"),),
            group_id="2",
            split="val",
        )
        predictions = [
            ValidationPrediction(sample=authentic, probability=np.zeros((8, 8), dtype=np.float32)),
            ValidationPrediction(sample=forged, probability=np.ones((8, 8), dtype=np.float32)),
        ]
        fake_gt = np.ones((8, 8), dtype=np.uint8)

        with patch("engine.validation_inference.load_resized_instance_masks", return_value=[fake_gt]):
            result = score_validation_predictions(
                predictions=predictions,
                pixel_util=None,
                pred_threshold=0.5,
                harden_temperature=1.0,
                hard_clip_low=0.0,
                hard_clip_high=1.0,
                min_component_area=0,
                compute_pixel_f1=False,
                verify_score_equivalence=False,
            )

        self.assertEqual(result["num_samples"], 2)
        self.assertEqual(result["num_authentic"], 1)
        self.assertEqual(result["num_forged"], 1)
        self.assertEqual(result["pixel_f1"], None)
        self.assertAlmostEqual(result["kaggle_score"], 1.0)

    def test_score_validation_predictions_requires_masks_when_pixel_f1_enabled(self) -> None:
        prediction = ValidationPrediction(sample=_sample("1"), probability=np.zeros((8, 8), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "gt_union_mask"):
            score_validation_predictions(
                predictions=[prediction],
                pixel_util=None,
                pred_threshold=0.5,
                harden_temperature=1.0,
                hard_clip_low=0.0,
                hard_clip_high=1.0,
                min_component_area=0,
                compute_pixel_f1=True,
                verify_score_equivalence=False,
            )
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_validation_inference -v
```

Expected:

```text
ImportError: cannot import name 'score_validation_predictions'
```

- [ ] **Step 3: Commit checkpoint if commits are approved**

```bash
git add tests/test_validation_inference.py
git commit -m "test: cover CPU validation scoring phase"
```

---

### Task 6: CPU Scoring Phase Implementation

**Files:**
- Modify: `engine/validation_inference.py`
- Test: `tests/test_validation_inference.py`

- [ ] **Step 1: Add CPU scoring imports**

Add these imports to `engine/validation_inference.py`:

```python
import time

from engine.validation_records import (
    build_solution_rows,
    build_submission_rows,
    compute_kaggle_score_from_instances,
    compute_kaggle_score_via_recodai,
    connected_components_to_masks,
    load_resized_instance_masks,
)
from inference.postprocess import post_process_prediction
from recodai_f1 import calculate_f1_score
```

- [ ] **Step 2: Add CPU scoring function**

Append this function to `engine/validation_inference.py`:

```python
def _post_process_probability(
    probability: np.ndarray,
    pixel_util,
    pred_threshold: float,
    harden_temperature: float,
    hard_clip_low: float,
    hard_clip_high: float,
    min_component_area: int,
) -> np.ndarray:
    probs = np.asarray(probability, dtype=np.float32)
    if pixel_util is None:
        pred_bin = (probs >= pred_threshold).astype(np.uint8)
        return pred_bin

    processed = post_process_prediction(
        probs=probs,
        pixel_util=pixel_util,
        threshold=pred_threshold,
        harden_temperature=harden_temperature,
        hard_clip_low=hard_clip_low,
        hard_clip_high=hard_clip_high,
        min_component_area=min_component_area,
    )
    return (processed >= pred_threshold).astype(np.uint8)


def score_validation_predictions(
    predictions: Sequence[ValidationPrediction],
    pixel_util,
    pred_threshold: float,
    harden_temperature: float,
    hard_clip_low: float,
    hard_clip_high: float,
    min_component_area: int,
    compute_pixel_f1: bool = False,
    verify_score_equivalence: bool = False,
) -> dict:
    postprocess_start = time.perf_counter()

    ordered_samples = [prediction.sample for prediction in predictions]
    pred_instances_by_sample_id: dict[str, list[np.ndarray]] = {}
    gt_instances_by_sample_id: dict[str, list[np.ndarray]] = {}
    shapes_by_sample_id: dict[str, tuple[int, int]] = {}
    pixel_f1s: list[float] = []

    for prediction in predictions:
        pred_bin = _post_process_probability(
            probability=prediction.probability,
            pixel_util=pixel_util,
            pred_threshold=pred_threshold,
            harden_temperature=harden_temperature,
            hard_clip_low=hard_clip_low,
            hard_clip_high=hard_clip_high,
            min_component_area=min_component_area,
        )
        sample = prediction.sample
        pred_instances_by_sample_id[sample.sample_id] = connected_components_to_masks(pred_bin)
        gt_instances_by_sample_id[sample.sample_id] = load_resized_instance_masks(sample, pred_bin.shape)
        shapes_by_sample_id[sample.sample_id] = pred_bin.shape

        if compute_pixel_f1:
            if prediction.gt_union_mask is None:
                raise ValueError("gt_union_mask is required when compute_pixel_f1=True")
            gt_bin = (prediction.gt_union_mask >= 0.5).astype(np.uint8)
            pixel_f1s.append(calculate_f1_score(pred_bin, gt_bin))

    postprocess_seconds = time.perf_counter() - postprocess_start
    scoring_start = time.perf_counter()

    kaggle_score = compute_kaggle_score_from_instances(
        ordered_samples,
        pred_instances_by_sample_id,
        gt_instances_by_sample_id,
    )

    if verify_score_equivalence:
        solution = build_solution_rows(ordered_samples, gt_instances_by_sample_id, shapes_by_sample_id)
        submission = build_submission_rows(ordered_samples, pred_instances_by_sample_id)
        official_score = compute_kaggle_score_via_recodai(solution, submission)
        if abs(float(kaggle_score) - float(official_score)) > 1e-8:
            raise AssertionError(f"Score equivalence check failed: direct={kaggle_score}, official={official_score}")

    scoring_seconds = time.perf_counter() - scoring_start
    pixel_f1 = float(np.mean(pixel_f1s)) if compute_pixel_f1 else None

    return {
        "kaggle_score": float(kaggle_score),
        "pixel_f1": pixel_f1,
        "num_samples": len(ordered_samples),
        "num_forged": sum(1 for sample in ordered_samples if sample.label == "forged"),
        "num_authentic": sum(1 for sample in ordered_samples if sample.label == "authentic"),
        "postprocess_seconds": float(postprocess_seconds),
        "scoring_seconds": float(scoring_seconds),
    }
```

- [ ] **Step 3: Run validation inference tests**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_validation_inference -v
```

Expected:

```text
Ran 5 tests
OK
```

- [ ] **Step 4: Run compile check**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache PYTHONPYCACHEPREFIX=/tmp/pycache uv run python -m py_compile engine/validation_inference.py
```

Expected: command exits with status 0.

- [ ] **Step 5: Commit checkpoint if commits are approved**

```bash
git add engine/validation_inference.py tests/test_validation_inference.py
git commit -m "feat: score validation predictions after GPU inference"
```

---

### Task 7: Refactor Validation Loop To Three Phases

**Files:**
- Modify: `engine/validate_loop.py`
- Test: `tests/test_validation_inference.py`, `tests/test_validation_records.py`

- [ ] **Step 1: Replace validation loop orchestration**

Replace the contents of `engine/validate_loop.py` with:

```python
from __future__ import annotations

import time
from typing import Sequence

import torch
from torch.utils.data import RandomSampler
from tqdm import tqdm

from dataset_utils import SampleRecord
from engine.validation_inference import collect_validation_predictions, score_validation_predictions


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    val_loader,
    val_samples: Sequence[SampleRecord],
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
    inference_mode: str = "direct",
    probability_dtype: str = "float16",
    log_timing: bool = True,
) -> dict:
    if isinstance(val_loader.sampler, RandomSampler):
        raise ValueError(
            "val_loader must use shuffle=False: sample index mapping to val_samples requires sequential order"
        )

    inference_start = time.perf_counter()
    progress_loader = tqdm(val_loader, desc=f"epoch {epoch_idx + 1} val inference")
    predictions = collect_validation_predictions(
        model=model,
        val_loader=progress_loader,
        val_samples=val_samples,
        device=device,
        inference_mode=inference_mode,
        sliding_window_fn=sliding_window_fn,
        probability_dtype=probability_dtype,
        collect_masks=compute_pixel_f1,
    )
    inference_seconds = time.perf_counter() - inference_start

    result = score_validation_predictions(
        predictions=predictions,
        pixel_util=pixel_util,
        pred_threshold=pred_threshold,
        harden_temperature=harden_temperature,
        hard_clip_low=hard_clip_low,
        hard_clip_high=hard_clip_high,
        min_component_area=min_component_area,
        compute_pixel_f1=compute_pixel_f1,
        verify_score_equivalence=verify_score_equivalence,
    )
    result["inference_seconds"] = float(inference_seconds)
    result["validation_inference_mode"] = inference_mode
    result["probability_dtype"] = probability_dtype

    if compute_pixel_f1:
        print(f"[non-official] pixel F1 (epoch {epoch_idx + 1}): {result['pixel_f1']:.4f}")

    if log_timing:
        print(
            "Validation timing "
            f"(epoch {epoch_idx + 1}): "
            f"inference={result['inference_seconds']:.2f}s "
            f"postprocess={result['postprocess_seconds']:.2f}s "
            f"scoring={result['scoring_seconds']:.2f}s"
        )

    return result
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_validation_inference tests.test_validation_records -v
```

Expected:

```text
OK
```

- [ ] **Step 3: Run compile check**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache PYTHONPYCACHEPREFIX=/tmp/pycache uv run python -m py_compile engine/validate_loop.py engine/validation_inference.py
```

Expected: command exits with status 0.

- [ ] **Step 4: Commit checkpoint if commits are approved**

```bash
git add engine/validate_loop.py engine/validation_inference.py
git commit -m "refactor: split validation into inference and scoring phases"
```

---

### Task 8: Validation Config And Training Wiring

**Files:**
- Modify: `configs/baseline_config.py`
- Modify: `train_baseline.py`
- Test: compile checks

- [ ] **Step 1: Add validation performance config fields**

In `configs/baseline_config.py`, add these fields to `BaselineConfig` near the existing validation metric fields:

```python
    validation_inference_mode: str = "direct"
    validation_probability_dtype: str = "float16"
    validation_log_timing: bool = True
```

The relevant section should read:

```python
    compute_pixel_f1: bool = False
    verify_score_equivalence: bool = False
    validation_inference_mode: str = "direct"
    validation_probability_dtype: str = "float16"
    validation_log_timing: bool = True
    checkpoint_dir: str = "runs/checkpoints"
```

- [ ] **Step 2: Pass config fields into validation**

In `train_baseline.py`, update the `validate_one_epoch(...)` call to include:

```python
            inference_mode=config.validation_inference_mode,
            probability_dtype=config.validation_probability_dtype,
            log_timing=config.validation_log_timing,
```

The end of the call should read:

```python
            compute_pixel_f1=config.compute_pixel_f1,
            verify_score_equivalence=config.verify_score_equivalence,
            inference_mode=config.validation_inference_mode,
            probability_dtype=config.validation_probability_dtype,
            log_timing=config.validation_log_timing,
        )
```

- [ ] **Step 3: Run compile check**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache PYTHONPYCACHEPREFIX=/tmp/pycache uv run python -m py_compile configs/baseline_config.py train_baseline.py
```

Expected: command exits with status 0.

- [ ] **Step 4: Run all non-training tests**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit checkpoint if commits are approved**

```bash
git add configs/baseline_config.py train_baseline.py
git commit -m "feat: default epoch validation to direct batched inference"
```

---

### Task 9: Human-Approved Timing Smoke Check

**Files:**
- Modify: none unless the human asks for config changes
- Test: human-approved training smoke only

- [ ] **Step 1: Ask for human approval before running training**

Ask:

```text
Do you want to run a tiny validation timing smoke check? This will initialize DINOv2, may download torch.hub weights if uncached, use GPU time, and may write a checkpoint under runs/checkpoints/.
```

- [ ] **Step 2: If approved, temporarily set tiny debug subsets**

In `configs/baseline_config.py`, temporarily set:

```python
    num_epochs: int = 1
    batch_size: int = 8
    train_subset: int | None = 8
    val_subset: int | None = 8
    validation_inference_mode: str = "direct"
    validation_probability_dtype: str = "float16"
    validation_log_timing: bool = True
```

- [ ] **Step 3: Run tiny training smoke only after approval**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python train_baseline.py
```

Expected:

```text
Validation timing (epoch 1): inference=...s postprocess=...s scoring=...s
Epoch 1: avg_loss=...  kaggle_score=...
```

- [ ] **Step 4: Restore normal defaults after the smoke check**

In `configs/baseline_config.py`, restore:

```python
    num_epochs: int = 10
    batch_size: int = 100
    train_subset: int | None = None
    val_subset: int | None = None
    validation_inference_mode: str = "direct"
    validation_probability_dtype: str = "float16"
    validation_log_timing: bool = True
```

- [ ] **Step 5: Re-run non-training tests after restoring defaults**

Run:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests -v
```

Expected:

```text
OK
```

- [ ] **Step 6: Commit checkpoint if commits are approved and config defaults are restored**

```bash
git add configs/baseline_config.py
git commit -m "test: smoke validation timing on tiny debug split"
```

---

## Review Checklist

- [ ] `validation_inference_mode="direct"` is the default.
- [ ] Exact-size 448x448 validation images do not use four sliding-window crops.
- [ ] Validation does not move masks to GPU unless `compute_pixel_f1=True`.
- [ ] GPU inference happens over loader batches in direct mode.
- [ ] CPU post-processing happens after probability collection.
- [ ] `kaggle_score` is still computed every epoch.
- [ ] Best checkpoint selection in `train_baseline.py` still uses `kaggle_score`.
- [ ] `verify_score_equivalence=True` still checks direct score against `recodai_f1.score`.
- [ ] Timing output reports inference, postprocess, and scoring seconds.
- [ ] All `unittest` tests pass without running training.

## Follow-Up Plans

Create separate plans for these items after this validation refactor is reviewed:

- `docs/colab_training_plan.md`: notebook setup for uploaded data, GPU verification, dependency handling without unnecessary Torch reinstall, and Drive checkpoint persistence.
- `docs/dino_model_size_plan.md`: evaluate DINOv2 Small versus Base, including `dino_embed_dim`, checkpoint compatibility, and score/speed comparison.
- `docs/high_resolution_validation_plan.md`: validate or infer at larger image sizes using optional sliding mode, crop batching, and optional global/local fusion.
