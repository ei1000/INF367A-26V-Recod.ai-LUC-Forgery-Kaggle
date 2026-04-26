# Implementation Plan

Deadline: 2026-04-26 23:59. All steps below are ordered by dependency.

## Step 0 — Derived source/target masks  `generate_source_target_masks.py` — done

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
- If no component clears the threshold, assign the most changed component to target
  so every forged case has target supervision.
- Save clean component masks to `train_masks_source/{case_id}.npy` and
  `train_masks_target/{case_id}.npy`.

For forged cases without authentic pairs:
- Save full union mask as target and empty source mask for later analysis only.
- Do not use these target-only labels in the initial BusterNet training run.

Also write `data/train_masks_source_target_metadata.csv` with `case_id`, pixel counts,
component counts, thresholds, and status flags for audit/debugging.

Current generated counts:
- `2751` forged cases processed
- `2377` derived from authentic/forged pairs
- `374` target-only fallbacks without authentic pairs, excluded from initial training
- `0` empty target masks

Audit notebook added: `einar_busternet/explore_source_target_masks.ipynb`.

## Step 1 — Data layer  `dataset.py` — done

Create `BusterNetDataset` extending the existing `ForgeryDataset` pattern.

Differences from `ForgeryDataset`:
- Output label is a single integer class map `(H, W)` with values
  `{0=background, 1=target, 2=source}` rather than a binary float mask or one-hot mask.
- Read precomputed masks from `data/train_masks_target/` and `data/train_masks_source/`.
- Initial training dataset filters forged samples to `status == "derived_from_pair"` in
  `data/train_masks_source_target_metadata.csv`; the 374 target-only cases are excluded.
- Include authentic samples as all-background labels for oF1 false-positive control.
  For the initial run, keep only authentic samples whose case ID belongs to a
  `derived_from_pair` forged sample. Later experiments can decide whether to add all
  authentic samples.
- If derived masks are missing for a forged case: fail clearly and ask to run Step 0.
- Forged label map construction:
  `label_map[target_mask > 0] = 1`, then `label_map[source_mask > 0] = 2`.
  This follows the spec's `{0=background, 1=target, 2=source}` convention.
- Authentic images: all pixels → class 0.
- Follow the baseline preprocessing: image resize with bilinear interpolation, mask/label
  resize with nearest-neighbor interpolation, RGB image tensor `(3, H, W)`, optional
  DINO/ImageNet normalisation, label tensor `(H, W)` with dtype `torch.long`.

Implemented constructor:

```python
BusterNetDataset(
    samples,
    data_root="data",
    target_size=448,
    use_rgb=True,
    normalize_rgb=True,
    rgb_mean=(0.485, 0.456, 0.406),
    rgb_std=(0.229, 0.224, 0.225),
    metadata_path=None,
    allowed_forged_statuses=("derived_from_pair",),
    include_authentic=True,
    authentic_policy="paired_derived_only",
)
```

Key inputs: `SampleRecord`, `data_root`, `target_size`.

Focused tests:
- paired forged sample returns an image tensor and a long label map containing
  background/target/source classes
- authentic sample returns an all-zero long label map
- no-pair forged sample is excluded by default
- missing Step 0 masks raise a clear error
- resized label maps still contain only integer class IDs `{0, 1, 2}`

Implemented tests: `tests/test_busternet_dataset.py`.

## Step 2 — Model  `model.py` — done

Implemented classes:

**`SelfCorrelPercPooling`**: stateless module implementing the GPU-native correlation.
- Input: `(B, C, H, W)` — DINO features
- Output: `(B, nb_pools, H, W)` — top-k similarity percentile features
- No learnable parameters.
- Validate `nb_pools > 0`.
- Use `F.normalize(..., eps=1e-6)` before `bmm` so zero/near-zero feature vectors stay
  finite.
- Sort similarity scores descending and gather `nb_pools` evenly spaced percentile
  positions with `torch.linspace(0, L - 1, nb_pools).round().long()`.
- Keep the diagonal self-similarity term for now. It is constant evidence at each
  location; removing it is a later ablation, not the initial implementation.

**`DinoBusterNet`**: the full model.
- `__init__`: takes `encoder`, `embed_dim`, `nb_pools`, `freeze_encoder`.
- `from_official(...)`: classmethod mirroring `DinoSegmenter.from_official`.
- `forward(x)`: returns raw logits `(B, 3, H, W)`.
- Reuse the baseline DINO padding and `forward_features` logic so non-448 or
  sliding-window inputs still work.
- Deprecated custom Mani decoder kept in code as `_DeprecatedCustomManiGridDecoder`:
  a DINO-grid CNN (`embed_dim→512→256→128`) with no branch upsampling.
- Current Mani decoder: progressive BusterNet-style decoder
  `32x32 → 64x64 → 128x128` for 448 input:
  `embed_dim→512→256`, upsample, `256→192`, upsample, `192→128`.
- Mani auxiliary classifier: `Conv2d(128,1,3,padding=1)` for Stage 1 target-mask BCE.
- Deprecated custom Simi decoder kept in code as `_DeprecatedCustomSimiGridDecoder`:
  a grid-level CNN (`nb_pools→256→128→96`) with no branch upsampling.
- Current Simi decoder: progressive BusterNet-style decoder
  `32x32 → 64x64 → 128x128` for 448 input:
  `nb_pools→256→192`, upsample, `192→128`, upsample, `128→96`.
- Simi auxiliary classifier: `Conv2d(96,1,3,padding=1)` for Stage 1 union-mask BCE.
- Fusion: concatenate Mani and Simi decoder features plus their auxiliary logits into
  `(B, 226, 128, 128)` for 448 input, then apply the wider fusion head from the spec.
- Final output is bilinearly upsampled from the refined fusion resolution to the padded
  image size, then cropped back to the original input size, matching `DinoSegmenter`.
- Exposes `forward_branches(x)` returning full-resolution one-channel `mani_logits` and
  `simi_logits` for Stage 1 auxiliary losses.

**`BusterNetUnionWrapper`**: evaluation adapter.
- Wraps a trained `DinoBusterNet`.
- Converts softmax source+target probability into one-channel binary logits with
  `torch.logit(...clamp(1e-6, 1 - 1e-6))`.
- Lets baseline validation call `sigmoid(model(x))` unchanged.

Focused tests:
- `SelfCorrelPercPooling` returns `(B, nb_pools, H, W)` and finite values for normal and
  all-zero features
- percentile pooling works for `nb_pools=1` and `nb_pools>1`
- `DinoBusterNet` forward returns `(B, 3, H, W)` for an input whose size is divisible by
  14 and for one that requires padding/cropping
- branch output API returns two `(B, 3, H, W)` tensors for Stage 1
- frozen encoder stays in eval mode when `model.train()` is called
- `BusterNetUnionWrapper` sigmoid output matches `P(target)+P(source)`

Implemented tests: `tests/test_busternet_model.py`.

## Step 3 — Config  `config.py` — done

Dataclass `BusterNetConfig` with all hyperparameters. Carries the baseline-compatible
fields needed by the shared loaders, DINO setup, validation, post-processing, and
checkpointing code. Re-exports `seed_worker` and `set_seed` from the baseline config.

Fields:
- Baseline-compatible fields: `batch_size`, `seed`, `target_size`, `pred_threshold`,
  post-processing settings, workers, DINO model name/embed dim, AMP, sliding-window
  settings, split ratios, checkpoint cadence, etc.
- Stage schedule: `stage1_epochs`, `stage2_epochs`, `stage3_epochs`.
- Stage learning rates: `stage1_lr=1e-2`, `stage2_lr=1e-2`, `stage3_lr=1e-5`.
- Loss settings: `ce_class_weights=(0.3, 1.0, 1.0)`, `union_wrapper_eps=1e-6`.
- BusterNet/model fields: `nb_pools=100`, `freeze_dino_encoder=True`,
  `fusion_mode="three_class"` by default. `fusion_mode="binary_union"` selects the
  binary fusion ablation.
- Dataset fields: `metadata_path`, `allowed_forged_statuses=("derived_from_pair",)`,
  `include_authentic=True`, `authentic_policy="paired_derived_only"`.
- Step 0 audit fields: `diff_threshold`, `component_change_fraction` for reporting only;
  training reads the generated masks and metadata.
- Checkpointing: `checkpoint_dir = "einar_busternet/artifacts/checkpoints"`
- Results: `results_dir = "einar_busternet/artifacts/results"`
- Checkpoint names: `best.pt`, `last.pt`
- Convenience: `total_stage_epochs` property.

Default stage schedule:
- `stage1_epochs=15`
- `stage2_epochs=10`
- `stage3_epochs=10`

Implemented tests: `tests/test_busternet_config.py`.

## Step 4 — Training script  `train.py` — code done, run pending

Three-stage training following the BusterNet paper curriculum. Continue from the baseline
training/validation stack and make small generic extensions only when needed. Reuses from
project root:
- Stage 1 needs a BusterNet-specific training loop because it uses `forward_branches`
  and two optimizers.
- Stage 2/3 can reuse `engine/train_loop.py` → `train_one_epoch` with the selected
  fusion loss.
- `engine/validate_loop.py` → `validate_one_epoch` with `BusterNetUnionWrapper(model)`.
- `engine/checkpointing.py` → save/load checkpoints
- `datasets/splits.py` → `make_grouped_stratified_splits`
- `dataset_utils.py` → `list_labeled_samples`

**Stage 1** — independent branch pre-training (auxiliary binary tasks, LR=1e-3):
- Mani optimizer: `Adam(mani_decoder + mani_classifier, lr=1e-3)`
- Simi optimizer: `Adam(simi_decoder + simi_classifier, lr=1e-3)` because
  `SelfCorrelPercPooling` has no learnable parameters.
- Loss: BCE+soft-Dice — `mani_logits[:, 0]` on target mask, and `simi_logits[:, 0]`
  on source+target union mask. These are raw one-channel auxiliary logits; do not apply
  sigmoid before the loss functions.
- Only the 2377 forged cases with authentic pairs and derived source/target labels, plus
  their authentic counterparts as all-background negatives
- Runs for `config.stage1_epochs` epochs

**Stage 2** — freeze branches, train Fusion only (LR=1e-2):
- Freeze all Mani-Det and Simi-Det parameters. DINO remains frozen.
- Optimizer: `Adam(fusion.parameters(), lr=1e-2)`
- Loss: `CrossEntropyLoss(weight=[0.3, 1.0, 1.0])` for `three_class`, or
  BCE+soft-Dice on source+target union for `binary_union`.
- Only the 2377 paired forged cases plus their authentic counterparts for the initial
  run. The 374 no-pair cases are excluded because target-only labels would teach the
  model that source regions are background.
- Runs for `config.stage2_epochs` epochs

**Stage 3** — unfreeze branches + Fusion, end-to-end fine-tuning (LR=1e-5):
- Unfreeze Mani-Det + Simi-Det + Fusion (DINOv2 stays frozen)
- Optimizer: `Adam(all_trainable_params, lr=1e-5)`
- Same fusion loss as Stage 2; LR halved on plateau, early stop on patience
- Planned next experiment: keep the same fusion loss but add small persistent auxiliary
  branch losses during Stage 3:

  ```text
  stage3_loss =
      fusion_loss
    + 0.1 * mani_aux_target_loss
    + 0.1 * simi_aux_union_loss
  ```

  Reason: later checkpoints improve forged recall and small/tiny buckets, but lose some
  authentic precision. Small auxiliary losses may preserve Mani/Simi branch semantics
  while fusion adapts.
- Same paired-only data policy as Stage 2 for the initial run.
- Runs for `config.stage3_epochs` epochs

Validation during training uses `BusterNetUnionWrapper(model)` so the baseline validation
path scores `P(target)+P(source)`. Checkpoint saving: best by validation kaggle_score
(same criterion as baseline). Save to `artifacts/checkpoints/best.pt`.

Planned validation/checkpoint extension:
- Keep `best.pt` for official Kaggle/oF1 compatibility.
- Add `best_balanced.pt` using a fixed-postprocess validation metric:

  ```text
  authentic_f1 = mean pixel F1 over authentic validation samples
  forged_f1 = mean pixel F1 over forged validation samples
  balanced_score = harmonic_mean(authentic_f1, forged_f1)
  ```

- Reuse the existing validation predictions and the same post-processing settings as
  training. Do not run a threshold sweep inside training.
- Expected overhead is a small CPU post-processing/aggregation pass after validation,
  not extra DINO inference.

Implemented:
- `configure_trainable_parts(model, stage)` keeps DINO frozen and switches branch/fusion
  trainability for stages 1/2/3.
- `train_stage1_epoch(...)` uses `forward_branches`, two optimizers, raw logits, and
  BCE+soft-Dice without external sigmoid. Mani learns target; Simi learns source+target
  union because similarity detection is symmetric.
- Fusion consumes branch decoder features plus auxiliary logits so the final head sees
  both rich features and direct Mani/Simi evidence.
- Stage 2/3 use baseline `train_one_epoch` with the selected fusion loss.
- Planned Stage 3 auxiliary-loss experiment will need a BusterNet-specific Stage 3 loop,
  because it must call both `forward(...)` and `forward_branches(...)` for the same
  batch. Keep tensors on GPU and avoid per-batch CPU metric syncs.
- Validation uses baseline `ForgeryDataset` plus `BusterNetUnionWrapper`, so the score is
  binary union on the normal validation split. The wrapper collapses 3-class output or
  passes binary-fusion logits through.
- Training transfers each batch to GPU once with non-blocking copies when pinned memory is
  enabled, and loss logging syncs once per epoch.
- Validation defaults to `validation_transfer_mode="accumulate_gpu"` to avoid per-batch
  GPU→CPU probability transfers before CPU post-processing/scoring.
- CLI overrides support quick smoke runs, subsets, batch size, stage epochs, validation
  transfer mode, and checkpoint weight loading.
- Best/last checkpoints go under `einar_busternet/artifacts/checkpoints`.

Implemented tests: `tests/test_busternet_train.py`.

## Step 5 — Evaluation  `evaluate.py` — done

Wraps existing `collect_validation_predictions` and `score_validation_predictions`,
matching the baseline evaluation flow.

Training uses either the native 3-channel BusterNet model or the binary-fusion ablation.
Evaluation uses a tiny `BusterNetUnionWrapper` whose `forward` converts 3-class logits
into one-channel binary foreground logits:

```python
probs = busternet(x).softmax(dim=1)          # (B, 3, H, W)
forgery_prob = probs[:, 1:2] + probs[:, 2:3] # target + source
binary_logits = torch.logit(forgery_prob.clamp(1e-6, 1 - 1e-6))
```

For `fusion_mode="binary_union"`, the wrapper returns the model's one-channel logits
directly. The existing baseline validation code can then apply `sigmoid(binary_logits)`
and receive the correct union probability. Post-processing and oF1 scoring stay
unchanged. Results saved to `artifacts/results/eval_summary.json`.

Implemented:
- Loads trusted local BusterNet checkpoints from
  `einar_busternet/artifacts/checkpoints/best.pt` by default.
- Reconstructs `BusterNetConfig` from the checkpoint and supports small CLI overrides.
- Evaluates the validation split, not the reserved local holdout split.
- Requires `--allow-torch-hub` before reconstructing DINO-BusterNet via torch hub.
- Saves a JSON summary under `einar_busternet/artifacts/results/eval_summary.json`.
- Adds `evaluate_validation_diagnostics.ipynb` for validation split diagnostics:
  forged/authentic breakdowns, false-positive/false-negative tables, probability
  distributions, prediction plots, and CSV exports.

Implemented tests: `tests/test_busternet_evaluate.py`,
`tests/test_busternet_evaluate_notebook.py`.

## Step 6 — README  `README.md`

Method description for grading. Cover:
- What BusterNet is and why it suits CMFD
- Architecture (mani + copy-det branches, self-similarity)
- Source/target derivation insight
- How it integrates with the group project
- Results (fill after eval)

## Step 7 — Progressive Mani/Simi decoding and higher-resolution fusion — implemented

Current decoder/fusion is useful but still too grid-like:
- DINO produces a low-resolution token grid.
- Mani and Simi decoders operate on that grid.
- Fusion also operates on that grid.
- Only the final logits are bilinearly upsampled to image size.

This is efficient, but it may be holding back small/tiny forged masks. The auxiliary
branch classifiers and fusion head are forced to decide boundaries and weak copied-source
regions before any spatial refinement has happened. Original BusterNet did the opposite:
it repeatedly decoded/upsampled branch features before classification and fusion.

Implemented ablation: keep DINO frozen, but replace the grid-only branch decoders with
progressive decoders. The old custom grid decoders remain in `model.py` with
`_DeprecatedCustom...` names for comparison/revert.

Recommended first version:

```text
DINO grid features: 32x32 for 448 input

Mani progressive decoder:
  32x32: 768 -> 512 -> 256
  upsample to 64x64
  64x64: 256 -> 192
  upsample to 128x128
  128x128: 192 -> 128
  mani_aux_logit: 128 -> 1

Simi progressive decoder:
  32x32: nb_pools -> 256 -> 192
  upsample to 64x64
  64x64: 192 -> 128
  upsample to 128x128
  128x128: 128 -> 96
  simi_aux_logit: 96 -> 1

Higher-resolution fusion:
  concat at 128x128:
    mani_features 128
    simi_features 96
    mani_aux_logit 1
    simi_aux_logit 1
    = 226 channels

  fusion:
    Conv 226 -> 160
    BN/ReLU
    Conv 160 -> 128
    BN/ReLU
    Conv 128 -> 64
    BN/ReLU
    Conv 64 -> output

Final:
  upsample logits from 128x128 to image size and crop padded borders.
```

Why this is more principled:
- Closer to BusterNet's repeated mask-decoder upsampling before classification/fusion.
- Matches modern dense-prediction patterns such as DPT/SegFormer-style feature
  reassembly/fusion better than only widening low-res convs.
- Better suited to medical/small-object segmentation, where coarse grids can erase thin
  structures and small duplicated regions.
- Still keeps the DINO backbone and training/evaluation pipeline unchanged.

Implementation notes:
- Start with bilinear upsampling + Conv/BN/ReLU blocks. Avoid transposed conv first.
- Keep auxiliary heads at the fusion resolution so Stage 1 trains spatially refined
  Mani/Simi maps, not only coarse token-grid maps.
- Keep binary union fusion as the main competition path.
- Add shape tests for 448 and non-divisible input sizes.
- Existing checkpoints will not load after this architecture change.

Later, if time allows, use DINO intermediate layers:
- collect several transformer block outputs,
- project each to a common channel width,
- reassemble/fuse them in DPT/SegFormer style.

This is more principled but touches the encoder feature path more, so progressive branch
decoding is the safer first ablation.

## Step 8 — Later fusion ablation — planned, not next

The current fusion module is capable and should not be changed before the Stage 3
auxiliary-loss experiment. If forged/source coverage still lags after that, test a
BusterNet-style multi-kernel fusion block while keeping binary union output:

```text
input: Mani features + Simi features + Mani aux logit + Simi aux logit
parallel convs: 1x1, 3x3, 5x5
concat + BN/ReLU
3x3 classifier to one-channel binary union logits
```

Reason: original BusterNet used BN-Inception-style fusion, and multi-kernel fusion can
mix local evidence with broader context. This is a later ablation because the current
results show fusion can already use the progressive decoder features; the immediate
failure mode is the authentic-vs-forged recall tradeoff during fine-tuning.

## File Map

```
einar_busternet/
├── SPEC.md               ← done
├── PLAN.md               ← done (this file)
├── README.md             ← write last (needs results)
├── __init__.py
├── generate_source_target_masks.py ← Step 0
├── dataset.py            ← Step 1 done
├── model.py              ← Step 2 done
├── config.py             ← Step 3 done
├── train.py              ← Step 4 code done
├── evaluate.py           ← Step 5 done
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
