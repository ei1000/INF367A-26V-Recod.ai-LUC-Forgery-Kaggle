# Handoff: BusterNet-DINO

This directory contains Einar's BusterNet-inspired individual project plan. The current
state is ready for Step 3 implementation.

## Current State

Steps 0, 1, and 2 are complete.

Generated artifacts:

- `data/train_masks_source/` — 2751 `.npy` masks
- `data/train_masks_target/` — 2751 `.npy` masks
- `data/train_masks_source_target_metadata.csv`

Metadata summary:

- `2377` cases have authentic/forged pairs and reliable source/target masks.
- `374` cases have no authentic pair and are marked `target_only_no_authentic`.
- Initial training uses forged rows with `status == "derived_from_pair"` plus their
  authentic counterparts as all-background negatives.

The 374 no-pair cases are intentionally reserved for later experiments. Do not use them
as target-only labels in the first training loop because that would teach the model that
source regions are absent/background.

## Important Decisions

Read these first:

- `einar_busternet/decisions_taken.md`
- `einar_busternet/PLAN.md`
- `einar_busternet/SPEC.md`
- `einar_busternet/DESIGN.md`

Key decisions:

- Use only the 2377 paired forged cases plus their authentic counterparts for initial
  BusterNet training.
- Use cosine similarity for DINOv2 features, not Pearson correlation.
- Use image difference only to classify clean GT connected components; do not use raw
  thresholded difference as the mask geometry.
- Model output is 3-channel logits `[background, target, source]`.
- Dataset label map should be `(H, W)` long tensor with classes
  `{0=background, 1=target, 2=source}`, batched as `(B, H, W)`.
- Evaluation wraps the 3-class model with `BusterNetUnionWrapper`, so the baseline
  validation path sees binary logits whose sigmoid is `P(target) + P(source)`.

## Files Added So Far

- `einar_busternet/source_target_masks.py`
  - Pure mask derivation logic.
  - No Matplotlib dependency.
- `einar_busternet/generate_source_target_masks.py`
  - CLI for Step 0 generation.
  - Run with:
    ```bash
    MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m einar_busternet.generate_source_target_masks --overwrite
    ```
- `tests/test_source_target_masks.py`
  - Unit tests for source/target split behavior.
- `visualization/forgery_plotter.py`
  - Matplotlib EDA helper; also updated with the same component-fallback behavior.
- `einar_busternet/dataset.py`
  - `BusterNetDataset`.
  - Filters forged cases to `derived_from_pair`.
  - Includes paired authentic samples as all-background labels.
  - Returns `(image, label_map)` where label map is `(H, W)` long with
    `{0=background, 1=target, 2=source}`.
- `tests/test_busternet_dataset.py`
  - Unit tests for filtering, label construction, resizing, and missing mask failures.
- `einar_busternet/model.py`
  - `SelfCorrelPercPooling`.
  - `DinoBusterNet`.
  - `BusterNetUnionWrapper`.
- `tests/test_busternet_model.py`
  - Unit tests for correlation pooling, model shapes, branch outputs, frozen encoder
    behavior, and evaluation wrapper probabilities.
- `einar_busternet/explore_source_target_masks.ipynb`
  - Lightweight visual audit notebook for paired and no-pair masks.

## Verification Already Run

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest tests.test_busternet_model tests.test_busternet_dataset tests.test_source_target_masks tests.test_forgery_plotter tests.test_forgery_dataset
```

Result:

```text
Ran 23 tests
OK
```

Post-generation invariants checked:

- source mask file count: `2751`
- target mask file count: `2751`
- metadata rows: `2751`
- target masks with zero pixels: `0`
- source + target pixel count mismatch vs original union: `0`

## Recommended Next Step

Implement Step 3 from `PLAN.md`:

- create `einar_busternet/config.py`
- add `BusterNetConfig`
- mirror baseline config fields needed for loaders, DINO, validation, checkpointing, and
  post-processing
- add BusterNet-specific fields:
  - `stage1_epochs`, `stage2_epochs`, `stage3_epochs`
  - `stage1_lr`, `stage2_lr`, `stage3_lr`
  - `nb_pools`
  - `ce_class_weights`
  - `union_wrapper_eps`
  - dataset filtering fields for `BusterNetDataset`

## Caution

Do not silently include the 374 `target_only_no_authentic` rows in training. They are
useful later, but using them now would create inconsistent supervision for a source/target
model.

Stage 1 uses `forward_branches(x)` and two optimizers, so it needs a small
BusterNet-specific training loop. Stage 2 and Stage 3 can reuse the baseline
`train_one_epoch` with `CrossEntropyLoss`.
