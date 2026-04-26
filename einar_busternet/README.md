# BusterNet-DINO — Individual Project (Einar)

## Transparency of AI tools
I used AI tools such as Codex while implementing this project. I did not outsource the research, design decisions, experiments, or interpretation. Most of the work went into understanding the task, planning the method, running experiments, reviewing code, and deciding what to keep.

The architecture is my own BusterNet adaptation based on the project baseline and modern dense-prediction practice. AI tools made it possible to implement and test ideas faster, but the important choices were made from inspection, experiments, and review.

Vibing and blindly trusting generated code would have most likely blown up my gpu. Identifying memory leaks and setting up proper implementation principles lead to efficient, readable and decent code. 

The workflow was: research, write a spec, write an implementation plan, implement one part at a time, test it, then inspect diagnostics before changing the next part. Then making a lot of hypotheses about the data, architecture, tricks and then testing it empirically with the implementations. 

## Method

This project implements a BusterNet-inspired dual-branch segmentation model for
copy-move forgery detection (CMFD), adapted to use frozen DINOv2 features as the
shared backbone.

### Background

BusterNet (Wu et al., ECCV 2018) was designed specifically for CMFD by recognizing
that copy-move forgery leaves two complementary signals in an image:

1. **Manipulation signal**: the pasted target region shows local artifacts and inconsistencies
   detectable by a standard segmentation branch (manipulation-detection branch).
2. **Copy-move signal**: the source and target regions are self-similar and contain
   the same visual content. A second branch that computes a self-similarity map can
   directly detect this paired structure. The original BusterNet uses Pearson correlation;
   this implementation uses cosine similarity since DINOv2 features are L2-normalized.

Combining both signals is more principled than binary segmentation alone for CMFD.

### Source/Target Label Derivation

The Kaggle dataset provides authentic-forged image pairs for 2377/2751 forged cases.
The absolute pixel difference between authentic and forged images cleanly separates
source from target:

- **Target** (pasted region): pixels show large intensity change (content was replaced).
- **Source** (copy origin): pixels are unchanged (original content remains in place).

Classification is done per connected component: if ≥25% of a component's pixels
changed by more than 5 intensity units, the component is labeled target; otherwise source.
Cases without an authentic pair are excluded from the source/target training loop because
there is no signal to identify which component is the target. Since the target region
shows local artifacts detectable by the Mani-Det branch, randomly attributing a component
as target would provide incorrect supervision.
This provides three-class ground truth (background / target / source) for training,
without requiring any additional annotation.

### Architecture

The model shares a frozen DINOv2 ViT-B/14 encoder with the project baseline, then
splits into two branches:

**Manipulation-detection branch**: A lightweight convolutional decoder applied directly
to the DINOv2 feature map, learning to detect local target/manipulation evidence.

**Copy-detection branch**: A `SelfCorrelPercPooling` module that computes a normalized
self-cosine-similarity matrix across all spatial locations in the DINOv2 feature map,
selects the top-k responses (percentile pooling), and passes the result through a
progressive convolutional decoder. This branch directly models the "same content appears
twice" structure of copy-move forgery.

The current final model uses progressive branch decoders, auxiliary branch classifiers,
and a multi-kernel fusion head. The submitted setting is `fusion_mode="binary_union"`:
the fusion head directly predicts the source+target union mask with BCE+Dice. This was
chosen for better problem alignment and empirically better performance. With binary_union
only the target auxiliary and the union GT mask are needed for training, not 3-class labels.
The 3-class path remains in code as the initial BusterNet-style experiment.

### Integration

The model integrates with the group project pipeline:

- Uses the same DINOv2 ViT-B/14 encoder and ImageNet normalization as the baseline.
- Uses the same train/validation splits (`make_grouped_stratified_splits`, seed=42).
- Produces a single-channel forgery probability map compatible with the shared
  post-processing and evaluation pipeline (`score_validation_predictions`).
- Checkpoints and results are self-contained in `einar_busternet/artifacts/`.
- Trained model weights: [best_balanced.pt](https://huggingface.co/ei1000/busternet-dino-cmfd-inf367a/blob/main/best_balanced.pt).

### Comparison with Baseline

This project directly tests whether explicit copy-move similarity modeling improves
over the baseline's single-branch approach, holding the feature extractor constant.

| | DINO Baseline | DINO BusterNet (this project) |
|---|---|---|
| Backbone | DINOv2 ViT-B/14 (frozen) | DINOv2 ViT-B/14 (frozen) |
| Branches | 1 (mani-det only) | 2 (mani-det + copy-det) |
| Self-similarity | No | Yes |
| Training labels | Binary union mask | Derived source/target auxiliaries + binary union fusion |
| Final objective | Binary segmentation | Binary union BCE+Dice |

## Results

| Metric | Value |
|---|---|
| Validation official score | 0.5335 |
| Validation authentic mean F1 | 0.8655 |
| Validation forged mean F1 | 0.3375 |
| Validation harmonic authentic/forged F1 | 0.4856 |
| Holdout official score | 0.5138 |
| Holdout authentic mean F1 | 0.8361 |
| Holdout forged mean F1 | 0.3292 |

The validation and holdout evaluation use the same baseline-style binary union task and
normal grouped splits as the DINO baseline. The source/target filter is only used for
BusterNet training, where clean branch labels are required.

## Running

```bash
# Train
uv run python -m einar_busternet.train --fusion-mode binary_union

# Evaluate on validation set
uv run python -m einar_busternet.evaluate --checkpoint einar_busternet/artifacts/checkpoints/best_balanced.pt --allow-torch-hub
```

## References

- Wu et al. "BusterNet: Detecting Copy-Move Image Forgery with Source/Target Localization."
  ECCV 2018.
- Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision." arXiv 2023.
