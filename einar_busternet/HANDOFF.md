# Handoff: BusterNet-DINO

This directory contains Einar's BusterNet-inspired individual project plan. The current
state is the final binary-union BCE+Dice direction with progressive branch decoders,
multi-kernel fusion, Stage 3 auxiliary loss, and balanced checkpoint tracking.

## Current State

Steps 0 through 5 are implemented. Recent experiments showed binary union fusion with
BCE+Dice is the current strongest direction.

Generated artifacts:

- `data/train_masks_source/` — 2751 `.npy` masks
- `data/train_masks_target/` — 2751 `.npy` masks
- `data/train_masks_source_target_metadata.csv`

Metadata summary:

- `2377` cases have authentic/forged pairs and reliable source/target masks.
- `374` cases have no authentic pair and are marked `target_only_no_authentic`.
- Initial training uses forged rows with `status == "derived_from_pair"` plus their
  authentic counterparts as all-background negatives.

The 374 no-pair cases are intentionally reserved for later training experiments. Do not
use them as target-only labels in the branch-training loop because that would teach the
model that source regions are absent/background.

This restriction is for training only. Evaluation and diagnostics use the baseline
`ForgeryDataset` on the normal grouped validation/holdout splits. They score binary union
masks and therefore include no-pair forged samples when those samples land in the split.

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
- Default model output is still 3-channel logits `[background, target, source]`.
- Binary fusion is the main current ablation with `fusion_mode="binary_union"` and
  outputs one-channel union logits.
- Current BusterNet post-processing defaults are `pred_threshold=0.2`,
  `min_component_area=10`, and `post_process_apply_opening=False`.
- Dataset label map should be `(H, W)` long tensor with classes
  `{0=background, 1=target, 2=source}`, batched as `(B, H, W)`.
- Evaluation wraps the model with `BusterNetUnionWrapper`, so the baseline validation
  path sees binary logits. The wrapper collapses 3-class output or passes binary-fusion
  logits through.

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
  - `BinaryFusionDinoBusterNet`.
  - `BusterNetUnionWrapper`.
- Fusion consumes Mani/Simi decoder features plus the two auxiliary branch logits.
- Branch auxiliary classifiers are explicit one-channel heads.
  - Current fusion input is 226 channels: 128 Mani features, 96 Simi features, and two
    one-channel auxiliary logits.
  - Mani/Simi decoders are progressive: DINO/SelfCorr grid features are refined at
    32x32, upsampled/refined to 64x64, then upsampled/refined to 128x128 before
    auxiliary classification and fusion.
  - The previous grid-only decoders are kept as `_DeprecatedCustom...` classes in
    `model.py`.
  - Old checkpoints from earlier fusion/decoder architectures will not load into this
    model.
- `tests/test_busternet_model.py`
  - Unit tests for correlation pooling, model shapes, branch outputs, frozen encoder
    behavior, and evaluation wrapper probabilities.
- `einar_busternet/config.py`
  - `BusterNetConfig`.
  - Baseline-compatible training/validation fields plus BusterNet stage, loss, dataset,
    and artifact settings.
  - Current stage schedule is `20 + 10 + 10` epochs.
  - Default `fusion_mode="three_class"`; use `"binary_union"` for the union-head ablation.
  - BusterNet validation defaults to `accumulate_gpu` transfer mode.
- `tests/test_busternet_config.py`
  - Unit tests for defaults, dataset policy, artifact paths, and baseline seed helpers.
- `einar_busternet/train.py`
  - Three-stage training entrypoint.
  - Stage 1 custom branch pretraining with two optimizers and raw BCE logits: Mani
    target mask, Simi source+target union mask.
  - Stage 1 trains decoder + auxiliary classifier for each branch.
  - Stage 2 reuses baseline `train_one_epoch` with 3-class CE or binary union BCE+Dice.
  - Stage 3 uses a BusterNet loop when auxiliary loss is enabled:
    `fusion_loss + 0.1 * mani_aux_target_loss + 0.1 * simi_aux_union_loss`.
  - Checkpointing keeps official `best.pt`, adds `best_balanced.pt` using
    harmonic authentic/forged validation F1 with fixed post-processing.
  - Validation wraps the model with `BusterNetUnionWrapper`.
  - Training loss logging syncs once per epoch, not once per batch.
  - Supports `--smoke` and small CLI overrides for first-run safety.
- `tests/test_busternet_train.py`
  - Unit tests for stage trainability and Stage 1 branch updates.
- `einar_busternet/evaluate.py`
  - Validation evaluation entrypoint for BusterNet checkpoints.
  - Uses `BusterNetUnionWrapper` and the shared validation scoring path.
  - Uses baseline `ForgeryDataset` on the normal validation split; no-pair forged cases
    are not filtered out for evaluation because evaluation only needs binary union masks.
  - Writes `einar_busternet/artifacts/results/eval_summary.json`.
- `tests/test_busternet_evaluate.py`
  - Unit tests for evaluation CLI guards and checkpoint config reconstruction.
- `einar_busternet/evaluate_validation_diagnostics.ipynb`
  - Notebook for validation diagnostics, forged/authentic breakdowns, and prediction
    plots.
- `tests/test_busternet_evaluate_notebook.py`
  - Unit test for the notebook diagnostics hooks.
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

Step 3 focused verification:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_busternet_config tests.test_busternet_model tests.test_busternet_dataset
```

Result:

```text
Ran 19 tests
OK
```

Step 4 focused verification:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_busternet_train tests.test_busternet_config tests.test_busternet_model tests.test_busternet_dataset
```

Result:

```text
Ran 21 tests
OK
```

Broader Step 4 verification:

```bash
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl uv run python -m unittest tests.test_busternet_train tests.test_busternet_config tests.test_busternet_model tests.test_busternet_dataset tests.test_source_target_masks tests.test_forgery_plotter tests.test_forgery_dataset tests.test_checkpointing
```

Result:

```text
Ran 39 tests
OK
```

Step 5 focused verification:

```bash
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl uv run python -m unittest tests.test_busternet_evaluate tests.test_busternet_train tests.test_busternet_config tests.test_busternet_model tests.test_busternet_dataset
```

Result:

```text
Ran 26 tests
OK
```

Latest focused verification after the postprocess/fusion ablation:

```bash
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl uv run python -m unittest tests.test_busternet_model tests.test_busternet_config tests.test_busternet_train
```

Result:

```text
Ran 27 tests
OK
```

Post-generation invariants checked:

- source mask file count: `2751`
- target mask file count: `2751`
- metadata rows: `2751`
- target masks with zero pixels: `0`
- source + target pixel count mismatch vs original union: `0`

## Recommended Next Step

Final run completed. Use `best_balanced.pt` for report diagnostics unless the report
specifically needs the official-score checkpoint:

```text
best_balanced.pt, epoch 37
validation official score: 0.5335
validation authentic mean F1: 0.8655
validation forged mean F1: 0.3375
holdout official score: 0.5138
holdout authentic mean F1: 0.8361
holdout forged mean F1: 0.3292
```

Command for retraining the final model if needed:

```bash
uv run python -m einar_busternet.train --fusion-mode binary_union
```

Then evaluate the new best checkpoint:

```bash
uv run python -m einar_busternet.evaluate --allow-torch-hub
```

This writes `einar_busternet/artifacts/results/eval_summary.json`.

## Caution

Do not silently include the 374 `target_only_no_authentic` rows in training. They are
useful later, but using them now would create inconsistent supervision for a source/target
model.

Stage 1 uses `forward_branches(x)` and two optimizers, so it needs a small
BusterNet-specific training loop. Stage 2 can reuse the baseline `train_one_epoch`.
Stage 3 uses the BusterNet-specific auxiliary-loss loop when `stage3_aux_loss_weight > 0`.

## Cleanup Plan

Behavior-preserving cleanup after the report:

- Remove deprecated grid-only decoders if the final architecture stays progressive.
- Move balanced validation metric code out of `train.py` into a shared helper used by
  the notebook and training.
- Consider dropping BCE-only binary loss helpers if no longer used.
- Add `balanced` checkpoint selection to `evaluate.py` for parity with the notebook.
- Normalize wording in docs from "ablation" to "final variant" once the final numbers
  are fixed.
- Archive old incompatible checkpoints to save disk space.
