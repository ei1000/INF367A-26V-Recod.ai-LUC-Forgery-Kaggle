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
`target_only_no_authentic`. They are reserved for later experiments, such as using a
trained model to infer likely source/target assignments or using them only for binary
union-mask fine-tuning after the source/target model is stable.

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
