# Decisions Taken

Main decisions behind BusterNet-DINO.

## Training Data And Evaluation Split

Training uses only forged cases with authentic counterparts:

- `2377` forged cases have reliable source/target masks from authentic-forged pairs.
- `374` forged cases do not have authentic pairs and are marked
  `target_only_no_authentic`.

The no-pair cases are not used for BusterNet training because target-only supervision
would teach the model that source regions are background. They remain available for a
future unknown-order experiment where duplicated components could be assigned to
source/target with a fixed seed or a permutation-invariant loss.

This filter applies only to training. Validation, diagnostics, and holdout use the
baseline `ForgeryDataset` on the normal grouped splits and score binary union masks.
That keeps the evaluation directly comparable to the baseline and includes no-pair
forged samples when they fall into the evaluated split.

## Source/Target Mask Derivation

The authentic-vs-forged difference is used as component evidence, not as the final mask.

- Split the Kaggle union mask into connected components.
- Mark a component as target if enough of its pixels changed from authentic to forged.
- Mark remaining components as source.
- If no component passes the threshold, assign the most changed component to target.

Reason: raw differences are noisy, especially around unchanged source regions. Clean
component masks from the Kaggle GT are better training labels.

## DINO Adaptations

The original BusterNet used VGG-16 and Pearson correlation. This version uses frozen
DINOv2 ViT-B/14 and cosine similarity.

- One shared frozen DINO encoder is enough because both branches would receive identical
  frozen features.
- Cosine similarity matches DINO's LayerNorm/self-distillation feature geometry better
  than z-scoring with Pearson correlation.
- `F.normalize(..., eps=...)` keeps self-correlation finite for degenerate vectors.

## Branch Semantics

The Mani branch is trained on target masks because pasted regions contain manipulation
evidence.

The Simi branch is trained on the source+target union because self-similarity is
symmetric. It should detect both duplicated regions, not infer historical source
direction. Earlier source-only Simi supervision was changed because it was conceptually
wrong and missed the purpose of similarity detection.

## Final Objective: Binary Union

The initial model kept the paper-like three-class fusion output:

```text
0 = background
1 = target
2 = source
evaluation = P(target) + P(source)
```

This was useful for comparison but hard to calibrate for the competition metric. Low
background weight caused authentic false positives; higher background weight made forged
predictions too empty.

The submitted path uses `fusion_mode="binary_union"`: branches still learn target and
union evidence, but final fusion predicts the union mask directly. This matches the
Kaggle/oF1 target while preserving BusterNet's Mani/Simi structure.

## Loss Function

Binary BCE alone was stable but too conservative on forged images. The final binary
loss is:

```text
loss = BCEWithLogitsLoss + dice_weight * SoftDiceLoss
```

BCE keeps pixel-wise probability learning stable and punishes authentic false positives.
Dice directly optimizes overlap and fights background dominance. The stable settings are:

```text
stage1_lr = 1e-3
branch_dice_weight = 0.5
fusion_dice_weight = 1.0
```

A first naive run used stronger Dice pressure with `stage1_lr=1e-2` and produced NaNs.
Lowering the Stage 1 LR and splitting branch/fusion Dice weights fixed that.

## Post-Processing Defaults

Diagnostics showed that opening and large component filtering removed useful small/tiny
forged predictions. The final defaults are:

```text
pred_threshold = 0.2
min_component_area = 10
post_process_apply_opening = False
```

This is judged by validation oF1 and forged/authentic diagnostics, not only by visual
inspection.

## Architecture Capacity

DINO dominates runtime, so adding capacity to branch decoders and fusion is cheap. The
final architecture keeps DINO frozen and expands only the BusterNet-specific heads.

Main changes tried and kept:

- Progressive Mani/Simi decoders: `32x32 -> 64x64 -> 128x128`.
- Auxiliary branch classifiers at the refined feature resolution.
- Fusion consumes decoder features plus Mani/Simi auxiliary logits.
- Fusion uses parallel `1x1`, `3x3`, and `5x5` conv branches, closer to BusterNet's
  BN-Inception fusion idea.

Reason: diagnostics showed weak forged/source/small-mask recall. Progressive decoding is
closer to BusterNet and modern dense prediction than classifying directly from the coarse
DINO token grid.

## Stage-Wise Training

The final curriculum is:

1. Stage 1: train Mani/Simi branches with auxiliary BCE+Dice losses.
2. Stage 2: freeze branches and train fusion.
3. Stage 3: unfreeze branches and fusion, keep DINO frozen.

Stage 3 keeps a small auxiliary branch term:

```text
stage3_loss =
    fusion_loss
  + 0.1 * mani_aux_target_loss
  + 0.1 * simi_aux_union_loss
```

Reason: later checkpoints improved forged localization but could lose some authentic
precision. Small auxiliary losses keep Mani target evidence and Simi union evidence
active while fusion adapts.

## Checkpoint Selection

Official `best.pt` is selected by Kaggle/oF1 and often prefers conservative authentic
behavior. Later checkpoints may localize forged masks better but trade away some
authentic F1.

Also saved:

```text
balanced_score = harmonic_mean(authentic_mean_f1, forged_mean_f1)
```

`best_balanced.pt` is not a replacement for official oF1 — it gives a clearer view of
whether the model actually localizes forged regions or just avoids predictions.

## Final Selected Result

Final numbers:

```text
best_balanced.pt, epoch 37
validation official score: 0.5335
validation authentic mean F1: 0.8655
validation forged mean F1: 0.3375
validation harmonic authentic/forged F1: 0.4856

holdout official score: 0.5138
holdout authentic mean F1: 0.8361
holdout forged mean F1: 0.3292
```

The baseline is strong at keeping authentic images clean. BusterNet-DINO adds
copy-move similarity and extra decoder/fusion capacity that improves forged localization.

