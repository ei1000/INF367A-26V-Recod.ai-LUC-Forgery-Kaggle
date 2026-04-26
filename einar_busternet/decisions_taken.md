# Decisions Taken

## Use Only Paired Cases for Initial BusterNet Training

The source/target split is reliable when a forged image has a matching authentic image.
For these cases, the authentic-vs-forged difference can identify which clean GT component
is the pasted target and which component is the stable source.

Current generated mask counts:

- `2377` forged cases have authentic pairs and derived source/target masks.
- `374` forged cases do not have authentic pairs.

For the initial BusterNet training run, use only the `2377` paired cases. The `374`
no-pair cases should not be used as target-only supervision at first.

Reason: the Kaggle union masks contain both source and target. Without an authentic pair,
we cannot reliably tell which clean component is source and which is target. If we label
the full union mask as target and leave source empty, we risk teaching the source branch
and fusion stage that source regions are background or not important. That could damage
the exact BusterNet behavior we are trying to exploit.

The `374` no-pair cases are still generated into `data/train_masks_target/` with empty
source masks and recorded in `data/train_masks_source_target_metadata.csv` as
`target_only_no_authentic`. They are reserved for later experiments, not Stage 1.

## Treat No-Pair Cases as Unknown-Order Pairs Later

Inspection of the no-authentic cases suggests that many mask instances already contain
both sides of a duplicated copy-move pair. In those cases, the historical direction is
ambiguous: either region could be called the source and the other the target without
changing the final competition objective, because evaluation uses the union
`source ∪ target`.

This does not recover true source/target history. Instead, it creates an
unknown-order source/target split that may provide more real copy-move supervision for
later training stages.

Planned policy:

- Stage 1 branch pretraining uses only clean paired data.
- No-pair cases may be considered after a first model is trained and inspected.
- If used, split each suitable no-pair mask instance into the two duplicated regions.
- Assign the two regions to source/target with a fixed random seed or use a
  permutation-invariant loss that accepts either assignment.
- Do not train on the current target-only fallback labels as if they were real
  source/target labels.

Reason: BusterNet's final prediction only needs `P(source) + P(target)`, so the exact
historical source/target direction is less important than learning that two related
foreground regions should both be localized. The risk is that arbitrary labels may weaken
branch semantics: the target branch may stop specializing in pasted artifacts or small
copy-move edits, while the source branch may stop specializing in stable donor regions.
For that reason, no-pair unknown-order supervision should be a controlled later
experiment, ideally after checking what the clean paired model already infers on these
cases.

## Use Cosine Similarity Instead of Pearson for DINOv2

The original BusterNet paper used Pearson correlation over VGG features. We use cosine
similarity for DINOv2 features because DINOv2 is LayerNorm-based and trained around
normalised feature directions. Z-scoring DINOv2 features would likely distort the feature
geometry that makes DINO useful.

## Use Difference as Component Evidence, Not Mask Geometry

The raw absolute difference can be speckled, especially over source components that show
faint gray/noisy changes. The clean output masks should come from the original Kaggle GT
components. The authentic-vs-forged difference is only used to classify whole GT
components as source or target.

## Binary Fusion Ablation Added

Initial BusterNet training kept the native 3-class output
`{background, target, source}` and collapsed it to a binary union only for validation.
This is closest to the BusterNet source/target localization design, but validation
showed a tradeoff: low background weight created authentic false positives, while higher
background weight made the model predict mostly target-like regions and miss much of the
union.

We added a separate `fusion_mode="binary_union"` ablation. It keeps Mani-Det target
supervision and Simi-Det union supervision, but lets the final fusion head predict only
the union forgery mask with binary BCE. The competition only scores the union, so this is
a reasonable problem-specific variant. The 3-class model remains available as
`fusion_mode="three_class"`.

When inspecting the binary fusion model it performed better. It got high f1 on authentic but low on forged. I tried tuning but it did not work. Then thought about changing the loss function to something more relevant.

Did some research and found that DICE is a better loss function for pixel overlap that is important for the forgery f1. Think that the BCE is still important to punish false positives on authentic. Therefore combining them weighted instead. 

Follow-up: binary fusion with BCE alone was still too conservative on forged images.
We changed the binary objective to BCE+soft-Dice, split Dice weights for branch and
fusion losses, and lowered Stage 1 LR:

```python
stage1_lr = 1e-3
branch_dice_weight = 0.5
fusion_dice_weight = 1.0
```

The first Stage 2 validation after this change reached `kaggle_score=0.5354`, the best
BusterNet result so far. This supports the decision that the competition-facing head
should optimize binary union overlap directly, not only per-pixel BCE or 3-class
source/target separation.

## Fuse Decoder Features, Not Auxiliary Branch Logits

The BusterNet paper feeds Mani-Det and Simi-Det mask-decoder feature maps into Fusion.
The auxiliary binary classifiers are only used for branch pretraining.

We updated the model to match that flow:

- Mani decoder outputs a feature map; a separate one-channel classifier predicts target.
- Simi decoder outputs a feature map; a separate one-channel classifier predicts union.
- Fusion consumes the concatenated decoder feature maps and predicts the 3-class mask.

This is an architecture-breaking change for old checkpoints. Retrain before evaluating
new results with this model definition.

## Loosen Small-Mask Post-Processing

Diagnostics showed that forged recall is weakest on small and tiny masks. Some forged
samples also produced isolated confident pixels that were removed before scoring. For the
next BusterNet runs we lowered the component filter and stopped opening by default:

```python
pred_threshold = 0.2
min_component_area = 10
post_process_apply_opening = False
```

Reason: opening and a `50` pixel component cutoff can erase thin or small copy-move
predictions. This may cost some authentic precision, so it should be judged by validation
oF1 and forged/authentic diagnostics, not only by visual inspection.

## Larger Fusion With Auxiliary Logit Hints

After BCE+Dice and binary union fusion, the remaining failure mode was mostly forged
false negatives and weak source coverage. We kept DINO frozen and enlarged only the cheap
fusion head in this first capacity ablation:

```text
mani decoder features: 96
simi decoder features: 64
mani auxiliary logit: 1
simi auxiliary logit: 1
fusion input: 162

Conv 162 -> 128 -> 128 -> 64 -> output
```

This slightly relaxes the earlier "features only" decision. Fusion still receives rich
decoder features, but it also gets the two branch classifiers as low-dimensional hints:
Mani gives target evidence and Simi gives union/similarity evidence. This is an
architecture-breaking ablation; old checkpoints should not be reused with this model
definition.

## Wider Decoders For Source And Small-Mask Recall

Validation diagnostics after the larger fusion run showed strong authentic control and
good large/medium forged masks, but small and tiny forged cases were still missed often.
Because DINO dominates runtime, we expanded the cheap branch decoders:

```text
Mani decoder: 768 -> 512 -> 256 -> 128
Simi decoder: 100 -> 256 -> 128 -> 96
Fusion input: 128 + 96 + 1 + 1 = 226
Fusion: 226 -> 160 -> 128 -> 64 -> output
```

Reason: if the Simi branch under-represents weak copied-source evidence, fusion cannot
recover it later. Wider branch decoders add capacity where the current failure mode
appears, while keeping the frozen DINO backbone and the training/evaluation pipeline
unchanged. This is another architecture-breaking ablation.

## Plan Progressive Branch Decoders Before More Blind Width

The current wider model still performs most reasoning on the low-resolution DINO token
grid and upsamples logits only at the end. This is efficient, but it is not very close to
the original BusterNet decoder, which repeatedly upsamples branch feature maps before
classification and fusion.

Decision: the next architecture ablation should be progressive Mani/Simi decoding and
higher-resolution fusion, not only wider grid-level convs.

Planned shape:

```text
DINO grid: 32x32 for 448 input
branch decode: 32x32 -> 64x64 -> 128x128
auxiliary heads: predict at 128x128
fusion: concatenate Mani features, Simi features, Mani aux logit, Simi aux logit at 128x128
final logits: upsample from 128x128 to image size
```

Reason: small/tiny forged regions and thin medical structures can be weakened when all
branch supervision and fusion happen on the coarse token grid. Progressive decoding gives
the branch losses a spatially refined feature map to train, while keeping DINO frozen and
the binary union objective unchanged.

We are not fully abandoning final bilinear upsampling. The final output can still be
upsampled at the end, but it should come from a refined 128x128 feature map rather than
directly from the 32x32 DINO grid. That is a better compromise between BusterNet-style
decoding and DINO runtime.

Implemented as the next architecture-breaking ablation:

```text
Mani: DINO grid -> Conv/BN/ReLU -> upsample -> Conv/BN/ReLU -> upsample -> 128 channels
Simi: SelfCorr grid -> Conv/BN/ReLU -> upsample -> Conv/BN/ReLU -> upsample -> 96 channels
Fusion: concatenate 128 + 96 + Mani aux + Simi aux at the refined branch resolution
```

The previous grid-only custom decoders were kept in `model.py` as
`_DeprecatedCustomManiGridDecoder` and `_DeprecatedCustomSimiGridDecoder` so we can
compare or revert without losing the code. Existing checkpoints are not compatible with
the progressive decoder model.

## Track Balanced Validation Separately From Official oF1

Validation diagnostics showed a recurring checkpoint tradeoff:

- best-by-official-oF1 checkpoints often keep authentic images cleaner,
- later checkpoints often localize forged masks better, especially across size buckets.

Decision: keep official `best.pt`, but add a balanced checkpoint for assignment/report
analysis:

```text
balanced_score = harmonic_mean(authentic_mean_f1, forged_mean_f1)
```

This does not replace the competition score. It adds a scientific model-selection view
that does not let authentic empty predictions dominate the story. Use the same fixed
post-processing settings as training validation; do not run threshold sweeps inside the
training loop.

## Try Stage 3 Auxiliary Loss Before Fusion Redesign

The current fusion module is capable: progressive-decoder runs show stronger forged
localization without changing fusion. The next training experiment should therefore keep
fusion unchanged and add small auxiliary branch losses during Stage 3:

```text
stage3_loss =
    fusion_loss
  + 0.1 * mani_aux_target_loss
  + 0.1 * simi_aux_union_loss
```

Reason: Stage 3 improves forged recall but can weaken authentic precision. Small
auxiliary losses may keep Mani target evidence and Simi union evidence stable while the
fusion head adapts. A BusterNet-style multi-kernel fusion block remains a later ablation,
not the next change.
