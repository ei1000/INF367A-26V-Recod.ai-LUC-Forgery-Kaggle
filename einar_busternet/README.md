# BusterNet-DINO — Individual Project (Einar)

## Transparency of AI tools
I have been using AI tools like codex when implementing this project. I have not outsourced the thinking, drafting, research, designing, planning and testing. I have spent most of my time researching, creating specs and plans, experimenting and reflecting to improve the project. 

The architecture is my own adaptation of busternet based on the baseline and modern practices. The tools have given me the oppertunity to implement and test ideas at a speed that would not have been possible with normal time and work constraints. I fully understand what the code does, and have spent much time doing code-review and refactoring.

For instance it loves to use torch autocast nested and introduce memory leaks causing OOM on both ram and Vram if you are not carefull.

I used a spec, plan, implement, test stragegy. Where I first do research and create a spec. Then create a plan for how to implement the code. Then implement and test one step at a time to ensure everything works. 

I wanted to be transparent with this whilst showcasing how it can be used responsibly. I am a software engineer, and we use these tools at work. This way I could spend more time on machine learning instead of code plumbing. The results backs this up too. 

## Method

This project implements a BusterNet-inspired dual-branch segmentation model for
copy-move forgery detection (CMFD), adapted to use frozen DINOv2 features as the
shared backbone.

### Background

BusterNet (Wu et al., ECCV 2018) was designed specifically for CMFD by recognizing
that copy-move forgery leaves two complementary signals in an image:

1. **Manipulation signal**: the pasted target region shows local artifacts and inconsistencies
   detectable by a standard segmentation branch (manipulation-detection branch).
2. **Copy-move signal**: the source and target regions are *self-similar* — they contain
   the same visual content. A second branch that computes the image's self-cosine-similarity
   map can directly detect this paired structure.

Combining both signals is more principled than binary segmentation alone for CMFD.

### Source/Target Label Derivation

The Kaggle dataset provides authentic-forged image pairs for 2377/2751 forged cases.
The absolute pixel difference between authentic and forged images cleanly separates
source from target:

- **Target** (pasted region): pixels show large intensity change (content was replaced).
- **Source** (copy origin): pixels are unchanged (original content remains in place).

Classification is done per connected component: if ≥25% of a component's pixels
changed by more than 5 intensity units, the component is labeled target; otherwise source.
Cases without an authentic pair are reserved for later analysis rather than used in the
initial source/target training loop.
This provides three-class ground truth (background / target / source) for training,
without requiring any additional annotation.

### Architecture

The model shares a frozen DINOv2 ViT-B/14 encoder with the project baseline, then
splits into two branches:

**Manipulation-detection branch**: A lightweight convolutional decoder applied directly
to the DINOv2 feature map, learning to detect local forgery evidence.

**Copy-detection branch**: A `SelfCorrelPercPooling` module that computes a normalized
self-cosine-similarity matrix across all spatial locations in the DINOv2 feature map,
selects the top-k responses (percentile pooling), and passes the result through a
convolutional decoder. This branch directly models the "same content appears twice"
structure of copy-move forgery.

The branch outputs are summed and decoded to a 3-channel prediction
[background, target, source], trained with weighted cross-entropy. At inference,
target and source probabilities are combined into a single forgery probability map
for evaluation with the competition metric.

### Integration

The model integrates with the group project pipeline:

- Uses the same DINOv2 ViT-B/14 encoder and ImageNet normalization as the baseline.
- Uses the same train/validation splits (`make_grouped_stratified_splits`, seed=42).
- Produces a single-channel forgery probability map compatible with the shared
  post-processing and evaluation pipeline (`score_validation_predictions`).
- Checkpoints and results are self-contained in `einar_busternet/artifacts/`.

### Comparison with Baseline

This project directly tests whether explicit copy-move similarity modeling improves
over the baseline's single-branch approach, holding the feature extractor constant.

| | DINO Baseline | DINO BusterNet (this project) |
|---|---|---|
| Backbone | DINOv2 ViT-B/14 (frozen) | DINOv2 ViT-B/14 (frozen) |
| Branches | 1 (mani-det only) | 2 (mani-det + copy-det) |
| Self-similarity | No | Yes |
| Training labels | Binary union mask | Derived source/target (3-class) |

## Results

*(To be filled after training and evaluation.)*

| Metric | Value |
|---|---|
| Validation oF1 | — |
| Authentic oF1 | — |
| Forged oF1 | — |

## Running

```bash
# Train
python -m einar_busternet.train

# Evaluate on validation set
python -m einar_busternet.evaluate
```

## References

- Wu et al. "BusterNet: Detecting Copy-Move Image Forgery with Source/Target Localization."
  ECCV 2018.
- Oquab et al. "DINOv2: Learning Robust Visual Features without Supervision." arXiv 2023.
