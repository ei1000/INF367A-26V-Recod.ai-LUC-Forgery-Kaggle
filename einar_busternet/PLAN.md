# Implementation Plan

Deadline: 2026-04-26 23:59. All steps below are ordered by dependency.

## Step 0 — Derived source/target masks  `generate_source_target_masks.py`

Create the BusterNet-compatible masks once before model training.

Outputs under the existing data layout:

```
data/
├── train_masks/          # original Kaggle union masks
├── train_masks_source/   # derived source/donor masks
└── train_masks_target/   # derived target/pasted masks
```

For forged cases with authentic pairs:
- Load authentic image, forged image, and original union mask.
- Compute absolute authentic-vs-forged difference.
- Threshold with `diff_threshold = 5.0` to ignore faint source noise.
- Split the original union mask into connected components.
- Use the diff only to classify clean GT components:
  - target if at least `component_change_fraction = 0.25` of pixels changed
  - source otherwise
- Save clean component masks to `train_masks_source/{case_id}.npy` and
  `train_masks_target/{case_id}.npy`.

For forged cases without authentic pairs:
- Save full union mask as target.
- Save empty source mask.

Optional but recommended:
- Write `data/train_masks_source_target_metadata.csv` with `case_id`, pixel counts,
  component counts, thresholds, and status flags for audit/debugging.
- Add a small notebook/audit cell that visualizes random generated pairs before training.

## Step 1 — Data layer  `dataset.py`

Create `BusterNetDataset` extending the existing `ForgeryDataset` pattern.

Differences from `ForgeryDataset`:
- Output label is a single integer class map `(H, W)` with values
  `{0=background, 1=target, 2=source}` rather than a binary float mask or one-hot mask.
- Read precomputed masks from `data/train_masks_target/` and `data/train_masks_source/`.
- If derived masks are missing for a forged case: fail clearly and ask to run Step 0.
- Authentic images: all pixels → class 0.
- Mask resize uses nearest-neighbor interpolation.

Key inputs: `SampleRecord`, `data_root`, `target_size`.

## Step 2 — Model  `model.py`

Two classes:

**`SelfCorrelPercPooling`**: stateless module implementing the GPU-native correlation.
- Input: `(B, C, H, W)` — DINO features
- Output: `(B, nb_pools, H, W)` — top-k similarity percentile features
- No learnable parameters.

**`DinoBusterNet`**: the full model.
- `__init__`: takes `encoder`, `embed_dim`, `nb_pools`, `freeze_encoder`.
- `from_official(...)`: classmethod mirroring `DinoSegmenter.from_official`.
- `forward(x)`: returns raw logits `(B, 3, H, W)`.
- Mani decoder: mirrors `DinoTinyDecoder` but outputs 3 channels.
- Copy decoder: lightweight 3-conv-block CNN on top of `SelfCorrelPercPooling`.
- Fusion: element-wise sum of branch outputs before upsample.

## Step 3 — Config  `config.py`

Dataclass `BusterNetConfig` with all hyperparameters. Mirrors `BaselineConfig` pattern
so existing utilities (`seed_worker`, `set_seed`) work unchanged.

Fields:
- All standard training fields (epochs, lr, batch_size, seed, target_size, etc.)
- BusterNet-specific: `nb_pools`, mask paths, `diff_threshold`,
  `component_change_fraction`
- Checkpointing: `checkpoint_dir = "einar_busternet/artifacts/checkpoints"`
- Results: `results_dir = "einar_busternet/artifacts/results"`

## Step 4 — Training script  `train.py`

Three-stage training following the BusterNet paper curriculum. Reuses from project root:
- `engine/train_loop.py` → `train_one_epoch` (pass custom loss via argument)
- `engine/validate_loop.py` → `validate_one_epoch`
- `engine/checkpointing.py` → save/load checkpoints
- `datasets/splits.py` → `make_grouped_stratified_splits`
- `dataset_utils.py` → `list_labeled_samples`

**Stage 1** — independent branch pre-training (auxiliary binary tasks, LR=1e-2):
- Mani optimizer: `Adam(mani_decoder.parameters(), lr=1e-2)`
- Simi optimizer: `Adam(simi_decoder.parameters() + corr_pooling.parameters(), lr=1e-2)`
- Loss: `BCEWithLogitsLoss` — mani on target mask, simi on source mask
- Only cases with authentic pairs (source/target split available)
- Runs for `config.stage1_epochs` epochs

**Stage 2** — freeze branches, train Fusion only (LR=1e-2):
- Freeze all Mani-Det and Simi-Det parameters
- Optimizer: `Adam(fusion.parameters(), lr=1e-2)`
- Loss: `CrossEntropyLoss(weight=[0.1, 1.0, 1.0])`
- All cases (fallback label for no-pair cases: forged→class 1)
- Runs for `config.stage2_epochs` epochs

**Stage 3** — unfreeze branches + Fusion, end-to-end fine-tuning (LR=1e-5):
- Unfreeze Mani-Det + Simi-Det + Fusion (DINOv2 stays frozen)
- Optimizer: `Adam(all_trainable_params, lr=1e-5)`
- Same CrossEntropyLoss; LR halved on plateau, early stop on patience
- Runs for `config.stage3_epochs` epochs

Checkpoint saving: best by validation kaggle_score (same criterion as baseline).
Save to `artifacts/checkpoints/best.pt`.

## Step 5 — Evaluation  `evaluate.py`

Wraps existing `collect_validation_predictions` and `score_validation_predictions`.

The model forward returns 3-channel logits. Before passing to the scoring pipeline
(which expects single-channel binary probabilities), combine:

```python
probs = logits.softmax(dim=1)          # (B, 3, H, W)
forgery_prob = probs[:, 1, :, :] + probs[:, 2, :, :]  # target + source
```

Then pass `forgery_prob` as the probability map through existing post-processing
and scoring. Results saved to `artifacts/results/eval_summary.json`.

## Step 6 — README  `README.md`

Method description for grading. Cover:
- What BusterNet is and why it suits CMFD
- Architecture (mani + copy-det branches, self-similarity)
- Source/target derivation insight
- How it integrates with the group project
- Results (fill after eval)

## File Map

```
einar_busternet/
├── SPEC.md               ← done
├── PLAN.md               ← done (this file)
├── README.md             ← write last (needs results)
├── __init__.py
├── generate_source_target_masks.py ← Step 0
├── dataset.py            ← Step 1
├── model.py              ← Step 2
├── config.py             ← Step 3
├── train.py              ← Step 4
├── evaluate.py           ← Step 5
└── artifacts/
    ├── checkpoints/
    └── results/
```

## Reuse from project root (import, do not copy)

- `dataset_utils.py` — `list_labeled_samples`, `SampleRecord`, `load_image_from_path`
- `datasets/splits.py` — `make_grouped_stratified_splits`
- `engine/train_loop.py` — `train_one_epoch`
- `engine/validate_loop.py` — `validate_one_epoch`
- `engine/validation_inference.py` — `collect_validation_predictions`, `score_validation_predictions`
- `engine/checkpointing.py` — `load_checkpoint`, `save_checkpoint`
- `inference/postprocess.py` — `post_process_prediction`
- `util/pixelmapUtil.py` — `PixelMapUtil`
- `configs/baseline_config.py` — `set_seed`, `seed_worker`

## Expected training time

Frozen DINOv2 + small decoders → similar to baseline (5–15 min on 4080 Super for 10 epochs).
Correlation matrix at 32×32 adds ~5ms per batch — negligible.
