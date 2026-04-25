# Handoff: BusterNet-DINO

This directory contains Einar's BusterNet-inspired individual project plan. The current
state is ready for Step 1 implementation.

## Current State

Step 0 is complete.

Generated artifacts:

- `data/train_masks_source/` — 2751 `.npy` masks
- `data/train_masks_target/` — 2751 `.npy` masks
- `data/train_masks_source_target_metadata.csv`

Metadata summary:

- `2377` cases have authentic/forged pairs and reliable source/target masks.
- `374` cases have no authentic pair and are marked `target_only_no_authentic`.
- Initial training must use only rows with `status == "derived_from_pair"`.

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

- Use only the 2377 paired forged cases for initial BusterNet training.
- Use cosine similarity for DINOv2 features, not Pearson correlation.
- Use image difference only to classify clean GT connected components; do not use raw
  thresholded difference as the mask geometry.
- Model output is 3-channel logits `[background, target, source]`.
- Dataset label map should be `(H, W)` long tensor with classes
  `{0=background, 1=target, 2=source}`, batched as `(B, H, W)`.
- Evaluation combines `P(target) + P(source)` into one binary forgery probability map.

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

## Verification Already Run

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest tests.test_source_target_masks tests.test_forgery_plotter
```

Result:

```text
Ran 6 tests
OK
```

Post-generation invariants checked:

- source mask file count: `2751`
- target mask file count: `2751`
- metadata rows: `2751`
- target masks with zero pixels: `0`
- source + target pixel count mismatch vs original union: `0`

## Recommended Next Step

Implement Step 1 from `PLAN.md`:

- create `einar_busternet/dataset.py`
- implement `BusterNetDataset`
- read `data/train_masks_source_target_metadata.csv`
- filter forged cases to `status == "derived_from_pair"` for initial training
- load `data/train_masks_target/{case_id}.npy`
- load `data/train_masks_source/{case_id}.npy`
- create a single `(H, W)` integer label map:
  - `0` background
  - `1` target
  - `2` source
- authentic samples should produce all-background labels if included
- use nearest-neighbor resizing for masks
- add focused tests before moving to the model

## Caution

Do not silently include the 374 `target_only_no_authentic` rows in training. They are
useful later, but using them now would create inconsistent supervision for a source/target
model.
