# Baseline Refactor Plan

Status: draft plan, refined after repo inspection  
Source spec: `docs/baseline_refactor_spec.md`  
Rule: do not implement code changes from this plan until the plan is approved.

## Spec Alignment Check

This plan follows the spec order and keeps the first implementation chunk focused on metric/data correctness:

1. Add sample metadata and split handling.
2. Add authentic-sample dataset support.
3. Add full `recodai_f1.score` validation.
4. Select checkpoints by full `kaggle_score`.
5. Keep `pixel_f1` as opt-in debug/fallback only.
6. Mark deprecated files after the core behavior works.

The plan adds one implementation-level refinement discovered from the local data layout: many stems exist in both `train_images/forged` and `train_images/authentic`. Because of that, splitting must be group-aware by stem to avoid placing a forged image in one split and its authentic counterpart in another split. This is consistent with the spec's stratification requirement and is a best-practice leakage guard, not a separate workstream.

Documentation rewrite, README cleanup, Kaggle-notebook-inspired inference improvements, decoder changes, TTA, global/local fusion, and threshold tuning remain out of scope for this core chunk.

## Goals

Implement the core baseline refactor from the spec:

1. Use all labeled training data with an 80/10/10 grouped and stratified train/validation/local-test split.
2. Support authentic samples during training and validation.
3. Validate with full `recodai_f1.score` every epoch.
4. Select and checkpoint the best model by full `kaggle_score`.
5. Keep `pixel_f1` only as an opt-in fallback/debug metric.
6. Preserve current repo post-processing defaults for the first implementation.
7. Keep the Kaggle notebook as inference inspiration, not a training rewrite.

## Non-Goals

- Do not change model architecture in the core refactor.
- Do not switch DINO loading from `torch.hub` to Hugging Face for training.
- Do not implement Kaggle notebook Sobel/global-local/TTA post-processing in the core metric/data refactor.
- Do not rewrite `docs/pipeline_suggestion.md` or the root README until code behavior is stable.
- Do not use supplemental/unlabeled/test-only images for training loss or validation scoring.
- Do not tune thresholds, class weighting, sampling, or decoder structure in this implementation chunk.
- Do not run local holdout test as part of normal training/model selection.

## Repository Reality Notes

The inspected repo currently behaves as follows:

- `train_baseline.py` is the maintained baseline entrypoint.
- `train_baseline.py` currently discovers only `data/train_images/forged`.
- `train_baseline.py` currently truncates with `train_subset=200` and `val_subset=50`.
- `ForgeryDataset` currently accepts only case IDs and returns `(img, mask)`.
- `ForgeryDataset` currently calls `load_image(case_id)` and `load_union_mask(case_id)`.
- `dataset_utils.find_image_path(case_id)` searches forged before authentic, so it is ambiguous when the same stem exists in both directories.
- Local data currently contains 2751 forged images, 2377 authentic images, and 2751 mask files.
- There are 2751 unique stems across forged/authentic; all 2377 authentic stems also exist in forged, and 374 stems are forged-only.
- `validate_one_epoch` currently returns a single float pixel F1.
- `recodai_f1.score` aligns solution/submission rows by row order, not by merging on `row_id_column_name`.
- `recodai_f1.score` mutates the solution DataFrame by adding columns, so validation should pass copies.
- `recodai_f1.py` is the maintained import path. The duplicate `recodai-f1.py` should not be imported by new code.
- `pyproject.toml` does not include scikit-learn; split logic should be implemented with the standard library unless another dependency is already needed.

## Validation Resolution Decision

For the core refactor, epoch validation should keep the current training-loop validation resolution:

- validation images come from `ForgeryDataset`,
- images and union masks are resized to `BaselineConfig.target_size`,
- sliding-window validation runs on those resized tensors,
- prediction masks are scored at the resulting prediction shape.

This gives official-style metric representation and authentic/instance behavior without mixing in high-resolution final-inference work. It is full `recodai_f1.score`, but at the resized validation resolution.

Implementation implication:

- Ground-truth forged instance masks must be resized individually with nearest-neighbor interpolation to the same `score_shape` as the prediction before RLE encoding.
- The solution row `shape` should be `json.dumps([height, width])` for the score shape actually used.
- Authentic solution rows remain `annotation = "authentic"` and `shape = "authentic"`.
- Final original-resolution inference/submission behavior belongs to the later inference/submission workstream.

## Data Model

Add a small internal sample metadata representation used by splitting, datasets, validation, and checkpoint metadata.

Recommended fields:

- `sample_id: str`
- `case_id: str`
- `label: Literal["forged", "authentic"]`
- `image_path: Path`
- `mask_paths: tuple[Path, ...]`
- `group_id: str`
- `split: Literal["train", "val", "test"] | None`

Rules:

- `case_id` is the file stem, for example `10`.
- `group_id` should be the same stem as `case_id` for grouped splitting.
- `sample_id` must be unique across labels, for example `forged:10` and `authentic:10`.
- Do not use `case_id` alone as a validation row ID because forged/authentic pairs can share the same stem.
- Do not load a sample image by `case_id` alone when a `SampleRecord` is available. Use `sample.image_path`.
- Forged samples come from `data/train_images/forged/*.png`.
- Authentic samples come from `data/train_images/authentic/*.png`.
- Forged samples should have one or more mask files in `data/train_masks`.
- Authentic samples should have no mask files and should use all-zero masks for training.
- Full-score validation should preserve forged instance masks where possible, not only the union mask.
- Full-score validation should represent authentic ground truth as `annotation = "authentic"` and `shape = "authentic"`.

## Split Design

Use a grouped stratified split:

1. Build `SampleRecord` objects for all labeled forged and authentic images.
2. Group records by `group_id`/stem.
3. Assign each group a group type:
   - `paired` if it contains forged and authentic records,
   - `forged_only` if it contains only forged records,
   - `authentic_only` if that ever appears in another dataset layout.
4. Shuffle groups within each group type using `BaselineConfig.seed`.
5. Allocate each group type into 80/10/10 train/validation/local-test groups.
6. Expand groups back to samples.

This prevents paired-image leakage while keeping the forged/authentic ratio close to stable in each split.

Implementation details:

- Validate that `train_ratio + val_ratio + test_ratio` is approximately `1.0`.
- Use deterministic ordering before shuffling so results are reproducible.
- Log total counts, per-split counts, and per-split forged/authentic counts.
- Do not use the local test split in `train_baseline.py` training or model selection.
- If split persistence is added, save generated JSON under `runs/splits/...` and add `/runs/` to `.gitignore` if it is not already ignored.
- It is also acceptable for the first implementation to regenerate splits deterministically from the seed and fixed data layout.

## File-Level Plan

### 1. `dataset_utils.py`

Add or modify helpers:

- Add `SampleRecord` as a frozen dataclass.
- Add `list_labeled_samples(data_root: Path = DATA) -> list[SampleRecord]`.
- Update `find_image_path(case_id, label=None, data_root=DATA)` so label-specific lookup is possible.
- Add `load_image_from_path(path: Path) -> np.ndarray`.
- Keep `load_image(case_id)` only as a legacy convenience path, but avoid using it in the refactored dataset when `SampleRecord` exists.
- Keep `find_mask_paths(case_id)` and allow an optional `data_root`.
- Add `load_instance_masks(mask_paths_or_case_id) -> list[np.ndarray]`.
- Keep `load_union_mask(case_id)` for legacy/debug use.
- Add `load_union_mask_from_paths(mask_paths) -> np.ndarray`.
- Add `load_mask_or_empty(sample, image_shape)` or equivalent support for authentic samples.

Implementation details:

- Directory membership defines the class label. Do not infer authenticity from missing masks alone.
- For forged samples, fail loudly if no mask files exist.
- For authentic samples, `mask_paths` should be empty and the training mask should be all zeros.
- Loaded masks should be squeezed and converted to binary `uint8`.
- Mask path sorting should remain deterministic.

### 2. New `datasets/splits.py`

Add split utilities:

- `make_grouped_stratified_splits(samples, seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)`.
- `group_samples_by_id(samples)`.
- `split_groups_by_type(groups, ratios, seed)`.
- `count_samples_by_split_and_label(splits)`.

Recommended return shape:

```python
{
    "train": list[SampleRecord],
    "val": list[SampleRecord],
    "test": list[SampleRecord],
}
```

Best-practice details:

- Do not add scikit-learn only for splitting.
- Preserve grouped stems across splits.
- Return samples in deterministic order inside each split after expansion.
- Keep the local test split out of normal training and validation loops.

### 3. `datasets/forgery_dataset.py`

Modify `ForgeryDataset` to accept sample records rather than only forged case IDs.

Changes:

- Constructor should accept `samples: Sequence[SampleRecord]`.
- Optional backwards compatibility with string case IDs is acceptable, but new training code should pass records.
- Store `self.samples` and use `sample.image_path` for image loading.
- If `sample.label == "forged"`, load the union mask from `sample.mask_paths`.
- If `sample.label == "authentic"`, create an all-zero mask matching the loaded image spatial shape.
- Keep current resize, RGB replication, normalization, and mask thresholding behavior for continuity.
- Keep the default return as `(img, mask)` so `engine/train_loop.py` does not need to change.

Optional metadata:

- Avoid returning `Path` objects through the default DataLoader collate unless needed.
- If metadata is added, gate it behind `return_metadata=True` and keep fields simple.
- The preferred validation path is to pass `val_samples` separately to `validate_one_epoch` and rely on `shuffle=False` to match DataLoader order.

### 4. New `engine/validation_records.py`

Add helpers for full-score validation row construction.

Recommended functions:

- `resize_binary_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray`
- `load_resized_instance_masks(sample: SampleRecord, shape: tuple[int, int]) -> list[np.ndarray]`
- `connected_components_to_masks(mask: np.ndarray) -> list[np.ndarray]`
- `prediction_mask_to_annotation(mask: np.ndarray) -> str`
- `solution_row_for_sample(sample: SampleRecord, shape: tuple[int, int]) -> dict`
- `build_solution_rows(ordered_samples, shapes_by_sample_id) -> pd.DataFrame`
- `build_submission_rows(ordered_samples, annotations_by_sample_id) -> pd.DataFrame`
- `compute_kaggle_score(solution, submission) -> float`

Rules:

- Authentic solution row: `sample_id`, `annotation = "authentic"`, `shape = "authentic"`.
- Forged solution row: `sample_id`, `annotation = rle_encode(instance_masks)`, `shape = json.dumps([height, width])`.
- Authentic prediction or empty prediction: `sample_id`, `annotation = "authentic"`.
- Non-empty prediction: connected components to instance masks, then semicolon-separated JSON RLE via `recodai_f1.rle_encode`.
- Prediction components should be extracted after current post-processing and component filtering.
- Solution and submission rows must be generated from the same `ordered_samples` list because `recodai_f1.score` currently aligns by row order.
- Call `recodai_f1.score(solution.copy(), submission.copy(), row_id_column_name="sample_id")`.

Important compatibility detail:

- `recodai_f1.rle_encode` expects a list of masks and returns semicolon-separated JSON RLE strings.
- `recodai_f1.rle_decode` in this repo accepts `authentic`; the duplicate `recodai-f1.py` should not be used.

### 5. `engine/validate_loop.py`

Refactor validation to compute full `kaggle_score`.

Signature direction:

```python
def validate_one_epoch(
    model,
    val_loader,
    val_samples,
    device,
    sliding_window_fn,
    pixel_util,
    pred_threshold,
    harden_temperature,
    hard_clip_low,
    hard_clip_high,
    min_component_area,
    epoch_idx,
    compute_pixel_f1=False,
) -> dict:
    ...
```

Changes:

- Require `val_loader` to use `shuffle=False`.
- Track sample index while iterating batches and map each tensor back to `val_samples[index]`.
- Run the current sliding-window prediction path.
- Keep current post-processing defaults:
  - `torch.sigmoid(logits)`,
  - `post_process_prediction`,
  - `pred_threshold`,
  - hardening config,
  - `min_component_area`.
- Convert post-processed binary masks to prediction annotations.
- Build solution and submission DataFrames in the same sample order.
- Call full `recodai_f1.score`.
- Return a structured validation result:

```python
{
    "kaggle_score": float,
    "pixel_f1": float | None,
    "num_samples": int,
    "num_forged": int,
    "num_authentic": int,
}
```

Optional pixel F1:

- Add a config-controlled opt-in path for `pixel_f1`.
- If enabled, compute it as a debug/fallback metric using union masks already returned by the validation dataset.
- Logs must clearly label it as non-official.

### 6. `configs/baseline_config.py`

Add or modify config fields:

- `data_root: str = "data"`
- `train_ratio: float = 0.8`
- `val_ratio: float = 0.1`
- `test_ratio: float = 0.1`
- `seed: int = 42` already exists.
- `train_subset: int | None = None`
- `val_subset: int | None = None`
- `compute_pixel_f1: bool = False`
- `checkpoint_dir: str = "runs/checkpoints"`
- `best_checkpoint_name: str = "best_by_kaggle_score.pt"`

Existing subset behavior:

- Normal baseline runs should use all split samples.
- Keep subset fields only as explicit debug controls.
- If debug subsets are used, apply them after splitting and make the log say this is a debug run.

### 7. `train_baseline.py`

Refactor the orchestration:

- Remove normal use of `get_forged_case_ids()` and `split_ids()`.
- Call `list_labeled_samples(Path(config.data_root))`.
- Call `make_grouped_stratified_splits(...)`.
- Use train split for `train_loader`.
- Use validation split for validation.
- Do not train or validate on local test split.
- Print/log split counts by label and group type.
- Instantiate `ForgeryDataset(train_samples, ...)` and `ForgeryDataset(val_samples, ...)`.
- Keep `val_loader` with `shuffle=False`.
- Call the refactored `validate_one_epoch(..., val_samples=val_samples, compute_pixel_f1=config.compute_pixel_f1)`.
- Use `validation_result["kaggle_score"]` for scheduler and best checkpoint selection.
- Rename `best_f1` to `best_kaggle_score`.
- Save the best checkpoint with `torch.save` instead of only keeping `best_model_state` in memory.
- Include checkpoint metadata:
  - epoch,
  - `kaggle_score`,
  - optional `pixel_f1`,
  - seed,
  - config values,
  - split counts,
  - model name,
  - post-processing thresholds.

Potential scheduler detail:

- Current `ReduceLROnPlateau` uses `mode="max"` and can continue using `kaggle_score`.

### 8. `.gitignore`

If the implementation writes checkpoints or split files under `runs/`, add:

```text
/runs/
```

This keeps generated model artifacts out of version control.

### 9. Deprecation Markers

After metric/data behavior works:

- Add a clear top-level comment/docstring to `main.py` marking it deprecated.
- Add a markdown cell at the top of `main.ipynb` if notebook editing is practical.
- Mention `train_baseline.py` as the maintained baseline entrypoint.
- Do not delete these files in the core refactor.

### 10. Optional Later Script: `evaluate_baseline.py`

Do not include this unless the core implementation is complete and the team wants a separate local holdout entrypoint in the same branch.

If added later:

- Load a checkpoint.
- Regenerate or load the same deterministic split.
- Evaluate the local test split once.
- Print `kaggle_score` and sample counts.
- Make the output clearly say that this is local holdout evaluation, not model-selection validation.

## Implementation Sequence

1. Add `SampleRecord`, labeled sample discovery, and mask/image helpers.
2. Add grouped stratified split utilities.
3. Update `ForgeryDataset` to use sample records and authentic all-zero masks.
4. Add validation row/RLE helpers.
5. Refactor validation to return full `kaggle_score`.
6. Refactor training orchestration to use all labeled data and 80/10/10 grouped splits.
7. Add checkpointing and model selection by `kaggle_score`.
8. Add optional `pixel_f1` flag and logging.
9. Add smoke tests / verification commands.
10. Mark deprecated files.
11. Review behavior before adding local holdout evaluation or final inference/submission work.

## Verification Plan

### Static Checks

- Targeted compile checks for changed modules:
  - `python3 -m py_compile dataset_utils.py`
  - `python3 -m py_compile datasets/forgery_dataset.py`
  - `python3 -m py_compile datasets/splits.py`
  - `python3 -m py_compile engine/validation_records.py`
  - `python3 -m py_compile engine/validate_loop.py`
  - `python3 -m py_compile train_baseline.py`
- Import checks for new helpers.

### Data Checks

Add a low-cost script or command path that prints:

- total labeled samples,
- forged count,
- authentic count,
- unique group/stem count,
- paired group count,
- forged-only group count,
- train/val/test counts,
- per-split forged/authentic counts,
- mask count consistency for forged samples.

Expected current local counts:

- forged images: `2751`
- authentic images: `2377`
- total labeled samples: `5128`
- unique stems/groups: `2751`
- paired groups: `2377`
- forged-only groups: `374`
- mask files: `2751`

### Unit-Like Smoke Checks

Use a tiny sample set without training:

- one paired forged/authentic group,
- one forged-only group if convenient,
- one fake empty prediction,
- one fake non-empty prediction.

Verify:

- `sample_id` is unique even when stems overlap,
- grouped split keeps matching stems in the same split,
- authentic training mask is all zeros,
- forged solution row uses resized instance-mask RLE and shape,
- authentic solution row uses `authentic`,
- empty prediction becomes `authentic`,
- non-empty prediction becomes RLE,
- solution/submission row order matches,
- `recodai_f1.score` runs without exception.

### Training Smoke Check

Run a very small debug subset only after the normal path is implemented:

- `num_epochs = 1`
- explicit debug train/val subset values
- CPU or CUDA depending availability

Expected:

- training loop starts,
- validation returns `kaggle_score`,
- checkpoint path is created,
- logs identify model selection by `kaggle_score`,
- logs identify `pixel_f1` as absent unless explicitly enabled.

## Risks And Mitigations

- **Risk:** full scoring is slower than pixel F1.  
  **Mitigation:** still run it each epoch per spec; keep optional debug subsets for development speed.

- **Risk:** paired authentic/forged stems leak across splits.  
  **Mitigation:** split by `group_id` first, then expand to samples.

- **Risk:** `case_id` alone loads the forged image for an authentic sample.  
  **Mitigation:** store and use `sample.image_path` everywhere in the refactored dataset.

- **Risk:** `recodai_f1.score` aligns rows by order.  
  **Mitigation:** build solution/submission rows from the same ordered sample list and pass copies to `score`.

- **Risk:** predicted union masks lose instance information.  
  **Mitigation:** split predictions into connected components before RLE.

- **Risk:** forged ground truth union masks understate official score behavior.  
  **Mitigation:** preserve individual mask files and resize each instance mask for full-score solution rows.

- **Risk:** authentic images dominate all-zero mask behavior.  
  **Mitigation:** keep sampling simple in first refactor, log per-class counts, revisit sampling only as a later measured experiment.

- **Risk:** local test split gets used during tuning.  
  **Mitigation:** keep local test out of `train_baseline.py` validation and reserve it for final local review/evaluation.

## Completion Criteria

The core refactor is complete when:

- `train_baseline.py` uses all labeled training data through an 80/10/10 grouped stratified split.
- Authentic samples train with all-zero masks.
- Paired forged/authentic stems stay in the same split.
- Forged samples preserve instance masks for full-score validation.
- Epoch validation reports full `kaggle_score`.
- Best checkpoint selection uses full `kaggle_score`.
- `pixel_f1` is opt-in only.
- The local holdout test split is not used during training or model selection.
- Deprecated files are clearly marked.
- Verification commands pass or any failures are documented.
