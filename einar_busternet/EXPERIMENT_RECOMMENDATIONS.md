# BusterNet Experiment Notes And Next Steps

Last updated: 2026-04-25.

This note tracks what we tried, what failed, what helped, and the next experiments.
Keep it practical. Details live in `SPEC.md`, `PLAN.md`, and `decisions_taken.md`.

## Current State

The best current direction is the binary-fusion BusterNet variant:

- Stage 1: Mani auxiliary head learns target mask.
- Stage 1: Simi auxiliary head learns source+target union mask.
- Stage 2/3: binary fusion head learns source+target union mask.
- Evaluation uses the normal baseline oF1 path.

Latest diagnostics for the binary-fusion run:

| Split | Empty rate | Mean predicted pixels | Mean pixel F1 | Mean max prob |
|---|---:|---:|---:|---:|
| authentic | 0.933 | 121 | 0.933 | 0.223 |
| forged | 0.659 | 2150 | 0.129 | 0.438 |

Interpretation:

- Authentic false positives are much better controlled than before.
- Forged recall is too low.
- The model often predicts nothing on forged samples.
- When it predicts, it often finds target-like regions but misses source regions.
- Post-processing is not the main issue yet; the raw probability maps are already too
  conservative on many forged images.

## What We Tried

### 3-Class Fusion, Low Background Weight

Setting:

```python
fusion_mode = "three_class"
ce_class_weights = (0.1, 1.0, 1.0)
```

Observed behavior:

- Simi=union made the model find more similar structures.
- But many authentic images produced false positives.
- Good forged recall sometimes came from over-predicting similar repeated structures.

Conclusion:

The fusion head was too willing to call similar structures forged. Background penalty was
too weak.

### 3-Class Fusion, Higher Background Weight

Settings tried:

```python
ce_class_weights = (0.5, 1.0, 1.0)
ce_class_weights = (0.3, 1.0, 1.0)
```

Observed behavior:

- `0.5` strongly fixed authentic false positives but made forged predictions too empty.
- `0.3` recovered forged recall, but authentic false positives came back.
- The 3-class softmax often seemed to learn the easiest foreground class/region and then
  relied on the wrapper to collapse source+target.

Conclusion:

The 3-class objective is not aligned enough with the competition union-mask score. It is
useful for source/target experiments, but not the most direct objective for oF1.

### Binary Fusion

Setting:

```python
fusion_mode = "binary_union"
```

Observed behavior:

- Authentic predictions improved compared with noisy 3-class settings.
- Forged predictions became conservative.
- Main problem is now false negatives, especially missing source regions.

Conclusion:

Binary fusion is the right direction for the competition objective, but BCE alone is not
enough for imbalanced segmentation. We need an overlap-aware loss.

## Loss Function Recommendation

Next experiment: use a compound loss for binary fusion and branch auxiliary heads.

Recommended first loss:

```text
loss = BCEWithLogitsLoss + soft Dice loss
```

Why:

- BCE gives stable pixel-wise probability training.
- Dice directly optimizes foreground/background overlap and is closer to pixel F1.
- Dice is less dominated by the many easy background pixels.
- This matches our current failure mode: authentic is mostly controlled, but forged
  recall/source coverage is weak.

Possible second loss:

```text
loss = BCEWithLogitsLoss + Tversky loss
```

Use this if Dice+BCE is still too conservative. Tversky can weight false negatives more
than false positives, which targets missing source/target pixels directly.

Suggested default order:

1. Binary fusion with `BCE + Dice`.
2. Stage 1 auxiliary heads also use `BCE + Dice`.
3. If forged empty rate remains high, try `BCE + Tversky` with higher false-negative
   penalty.
4. Only then tune prediction threshold/post-processing.

Implementation notes:

- Use logits as input.
- Compute probabilities inside the Dice/Tversky loss with `sigmoid`.
- Smooth numerator/denominator with a small epsilon to avoid division by zero.
- Average Dice over batch samples, not over the whole batch only, so tiny masks do not
  disappear inside large masks.
- Keep loss weights simple first:

```text
total = bce + dice
```

or, if Dice is too aggressive:

```text
total = bce + 0.5 * dice
```

## Diagnostics To Add

Cheap notebook additions:

- forged empty rate
- forged nonempty mean F1
- authentic nonempty rate
- GT-size buckets for forged samples:
  - tiny
  - small
  - medium
  - large
- mean recall and precision separately for forged nonempty samples
- count of forged samples with `max_prob` near threshold but removed by post-processing

Why:

- Current mean forged F1 hides two problems: empty predictions and partial masks.
- We need to know whether the loss helps the model fire more often or only improves masks
  after it fires.

## Architecture Recommendation

Do loss first. If loss improves but source coverage remains weak, enlarge fusion and feed
auxiliary logits into fusion.

Current binary fusion input:

```text
mani_features: 96 channels
simi_features: 64 channels
concat: 160 channels
```

Suggested next fusion input:

```text
mani_features: 96 channels
simi_features: 64 channels
mani_aux_logit: 1 channel
simi_aux_logit: 1 channel
concat: 162 channels
```

Suggested larger fusion body:

```text
Conv 162 -> 128
BN/ReLU
Conv 128 -> 128
BN/ReLU
Conv 128 -> 64
BN/ReLU
Conv 64 -> 1
```

Why this is cheap:

- DINO dominates runtime.
- Decoder/fusion compute is small compared with DINO.
- We can add capacity without changing baseline preprocessing or validation.

Why auxiliary logits may help:

- Mani aux logit gives direct target evidence.
- Simi aux logit gives direct union/similarity evidence.
- Decoder features carry richer information, but aux logits give the fusion head a clean
  low-dimensional hint.
- This may help recover source regions, which appear to be mostly available through the
  Simi branch.

## Strong Auxiliary Supervision: Why It May Help

Reasons to keep or strengthen auxiliary supervision:

- It prevents the Simi branch from becoming a passive feature extractor during Stage 3.
- It forces Simi to keep detecting both copied regions, not only the easiest target-like
  region.
- It gives gradients directly to each branch even if fusion initially ignores one branch.
- It matches BusterNet's design: branch classifiers are used to make Mani-Det and Simi-Det
  learn useful standalone masks before fusion.

Potential implementation:

```text
stage3_loss =
    fusion_binary_loss
  + 0.1 * mani_aux_target_loss
  + 0.1 * simi_aux_union_loss
```

Do not start here. First try better branch/fusion loss. Add persistent auxiliary loss
only if Stage 3 causes branch drift or source coverage remains weak.

## Threshold And Post-Processing

Threshold sweeps are useful for diagnosis but should not be the main fix yet.

Expected behavior:

- Lower threshold may recover forged masks.
- It will likely wake authentic false positives.
- If Dice+BCE increases raw forged probability without increasing authentic probability,
  threshold tuning becomes safer.

Recommended diagnostic sweep:

```text
thresholds = [0.35, 0.4, 0.45, 0.5, 0.55]
```

Keep the sweep small.

## Source Notes

Loss-function rationale is consistent with segmentation literature:

- Region-based losses such as Dice are widely used for imbalanced segmentation.
- Dice+CE/BCE compound losses are common because Dice improves overlap sensitivity while
  CE/BCE stabilizes probability training.
- Tversky changes the false-positive/false-negative tradeoff and can raise recall when
  false negatives dominate.

Useful sources:

- Hosseini and Baghshah, "Dilated Balanced cross entropy loss for medical image
  segmentation", BMC Medical Imaging, 2026.
  https://link.springer.com/article/10.1186/s12880-026-02245-y
- Abraham and Khan, "A Novel Focal Tversky loss function with improved Attention U-Net
  for lesion segmentation", arXiv, 2018.
  https://arxiv.org/abs/1810.07842
- Ma et al., "Loss odyssey in medical image segmentation", Medical Image Analysis, 2021.
  https://doi.org/10.1016/j.media.2021.102035
