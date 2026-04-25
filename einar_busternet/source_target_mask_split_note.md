# Source/Target Mask Split From Existing Data

The Kaggle masks in this project mark the full duplicated copy-move region. In other words, the ground truth is not only the pasted/forged target region: it contains both the source/donor region and the target/pasted region as one union mask.

This is useful for BusterNet-style training because BusterNet is designed around source and target localization. The main Kaggle metric does not require us to distinguish source from target, since both are foreground in the final prediction. Still, a model that explicitly learns both regions is directly relevant because the competition foreground already contains both regions.

## Intuition

We have paired images for many training cases:

- the original authentic image
- the forged copy-move version
- the union ground-truth mask containing source and target

The key observation is that the pasted target region changes between the authentic and forged image, while the source region is mostly stable. When looking at the absolute difference, the target tends to have strong colored/intensity changes, while the source often appears as faint gray noise or stays nearly unchanged.

So the difference image should not become the final mask directly. It is too noisy and produces speckled source/target maps. Instead, we use the clean original ground-truth mask for geometry, and use the image difference only to decide which clean component is source and which is target.

## Practical Split

The implemented approach is:

1. Load the authentic image, forged image, and original union mask.
2. Compute the absolute difference between authentic and forged.
3. Threshold the difference to ignore faint low-intensity source noise.
4. Split the union mask into connected components.
5. For each clean GT component, measure how much of it has strong image difference.
6. Components with enough changed pixels are labeled target.
7. The remaining GT components are labeled source.

This gives clean source and target masks because the output masks are whole components from the original ground truth, not raw thresholded difference pixels.

In code this is exposed through:

```python
from visualization import ForgeryDataPlotter

plotter = ForgeryDataPlotter("data")
masks = plotter.derive_source_target_masks(case_id)

source_mask = masks.source_mask
target_mask = masks.target_mask
union_mask = masks.union_mask
```

The component diagnostics are available as:

```python
masks.component_scores
```

This makes it possible to create BusterNet-compatible supervision from the existing competition data instead of generating a new synthetic dataset first.
