# BusterNet-DINO Implementation Plan

Compact implementation status for BusterNet-DINO.
Detailed experiment history lives in `decisions_taken.md`.

## Goal

Build a BusterNet-inspired copy-move forgery detector on top of the project DINOv2
baseline. Keep the frozen DINO backbone and baseline data/evaluation stack, then add:

- Mani-Det branch for pasted-target evidence.
- Simi-Det branch for internal copy-move similarity evidence.
- Progressive branch decoding before fusion.
- Binary union fusion for the Kaggle/oF1 objective.
- BCE+Dice losses for pixel overlap.

## Implemented Pipeline

### 1. Source/Target Masks

Implemented in `source_target_masks.py` and `generate_source_target_masks.py`.

- Use authentic-vs-forged difference to classify clean union-mask components as target or
  source.
- Save derived masks to `data/train_masks_target/` and `data/train_masks_source/`.
- Save audit metadata to `data/train_masks_source_target_metadata.csv`.
- Current counts: `2377` reliable `derived_from_pair` forged cases and `374`
  `target_only_no_authentic` cases reserved for later experiments.

The difference image is evidence only; final training masks come from the clean Kaggle
GT components.

### 2. Dataset

Implemented in `dataset.py`.

- `BusterNetDataset` returns `(image, label_map)`.
- `label_map` is a long tensor with `{0=background, 1=target, 2=source}`.
- Forged training samples are filtered to `status == "derived_from_pair"`.
- Authentic paired negatives are included as all-background labels.
- Image resize uses bilinear interpolation; label resize uses nearest neighbor.

### 3. Model

Implemented in `model.py`.

- Shared frozen DINOv2 ViT-B/14 feature extractor.
- `SelfCorrelPercPooling` computes cosine all-pairs similarity on the DINO token grid and
  keeps `nb_pools=100` percentile channels.
- Mani decoder progressively refines `32x32 -> 64x64 -> 128x128` features and predicts a
  one-channel target auxiliary mask.
- Simi decoder progressively refines self-correlation features and predicts a
  one-channel source+target union auxiliary mask.
- Fusion concatenates Mani features, Simi features, Mani aux logits, and Simi aux logits
  at `128x128`.
- Fusion head uses parallel `1x1`, `3x3`, and `5x5` conv branches, then a small
  classifier.
- `BinaryFusionDinoBusterNet` is the submitted path with one-channel union logits.
- `DinoBusterNet` with three-class output is kept for comparison with the initial
  BusterNet-style source/target experiment.
- `BusterNetUnionWrapper` adapts either output shape to the baseline binary evaluator.

### 4. Config

Implemented in `config.py`.

Current important defaults:

```text
stage1_epochs = 20
stage2_epochs = 10
stage3_epochs = 10
stage1_lr = 1e-3
stage2_lr = 1e-2
stage3_lr = 1e-5
branch_dice_weight = 0.5
fusion_dice_weight = 1.0
stage3_aux_loss_weight = 0.1
pred_threshold = 0.2
min_component_area = 10
post_process_apply_opening = False
```

Training command used for the final method:

```bash
uv run python -m einar_busternet.train --fusion-mode binary_union
```

### 5. Training

Implemented in `train.py`.

- Stage 1: train Mani and Simi branches with separate optimizers.
  - Mani target: `label == 1`
  - Simi target: `label > 0`
  - Loss: BCE+soft-Dice on raw logits.
- Stage 2: freeze branches, train fusion.
  - Binary method uses BCE+soft-Dice on union labels.
  - Three-class comparison uses weighted cross entropy.
- Stage 3: unfreeze branches and fusion, keep DINO frozen.
  - Main fusion loss plus `0.1` Mani/Simi auxiliary losses.
- Validation uses baseline `ForgeryDataset`, `BusterNetUnionWrapper`, and the normal
  grouped split. Evaluation is therefore directly comparable to the baseline and includes
  no-pair forged validation samples when they are in the split.
- Training saves:
  - `best.pt` by official validation oF1.
  - `best_balanced.pt` by harmonic mean of authentic and forged mean F1.
  - `last.pt`.

The training loop keeps tensors on GPU during forward/backward and only synchronizes
metrics at epoch/validation boundaries.

### 6. Evaluation And Diagnostics

Implemented in `evaluate.py` and `evaluate_validation_diagnostics.ipynb`.

- `evaluate.py` reconstructs the model from checkpoint config and evaluates the normal
  validation split.
- Diagnostics notebook supports `best`, `balanced`, `last`, and `custom` checkpoints.
- Notebook reports official score, authentic/forged breakdown, size buckets, threshold
  and post-processing sweeps, false positives, misses, successes, random examples, and
  optional holdout metrics.

## Final Report Numbers

Selected checkpoint:

```text
checkpoint: best_balanced.pt, epoch 37
validation official score: 0.5335
validation authentic mean F1: 0.8655
validation forged mean F1: 0.3375
validation harmonic authentic/forged F1: 0.4856
forged nonempty mean F1: 0.4955

holdout official score: 0.5138
holdout authentic mean F1: 0.8361
holdout forged mean F1: 0.3292
```

`best.pt` is best on official oF1. `best_balanced.pt` better shows forged localization
without collapsing authentic control.

## Reused Project Code

Imported instead of copied:

- `dataset_utils.py`
- `datasets/splits.py`
- `engine/train_loop.py`
- `engine/validate_loop.py`
- `engine/validation_inference.py`
- `engine/checkpointing.py`
- `inference/postprocess.py`
- `configs/baseline_config.py`

