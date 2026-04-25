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

## Fuse Decoder Features, Not Auxiliary Branch Logits

The BusterNet paper feeds Mani-Det and Simi-Det mask-decoder feature maps into Fusion.
The auxiliary binary classifiers are only used for branch pretraining.

We updated the model to match that flow:

- Mani decoder outputs a feature map; a separate one-channel classifier predicts target.
- Simi decoder outputs a feature map; a separate one-channel classifier predicts union.
- Fusion consumes the concatenated decoder feature maps and predicts the 3-class mask.

This is an architecture-breaking change for old checkpoints. Retrain before evaluating
new results with this model definition.
