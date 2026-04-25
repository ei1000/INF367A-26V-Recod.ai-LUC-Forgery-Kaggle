# BusterNet-DINO Specification

## Goal

Implement a BusterNet-inspired dual-branch copy-move forgery detection model that
replaces the original VGG-16 backbone with frozen DINOv2 features. This enables a
direct architectural comparison with the project baseline (DINOv2 + single mani branch),
isolating the contribution of explicit copy-move similarity modeling.

## Key Insight: Source/Target Label Derivation

The Kaggle dataset provides forged images and their authentic counterparts.
By computing the per-pixel absolute difference between forged and authentic images,
we can classify each connected component of the union forgery mask:

- **Target** (pasted region): component where ≥ `component_change_fraction` (default 25%)
  of pixels show intensity change > `diff_threshold` (default 5.0). The pasted content
  replaces original pixels, so these show high difference.
- **Source** (copy origin): remaining components. The source region is copied elsewhere
  but the original pixels remain in place, so the difference is low.

This derivation mirrors `visualization/forgery_plotter.py::derive_source_target_masks`
and is materialized before training into `data/train_masks_source/` and
`data/train_masks_target/`.
~86% of forged training cases (2377/2751) have authentic pairs and can use this labeling.
The remaining 374 cases have generated target-only fallback masks for later analysis,
but are excluded from the initial BusterNet training run.
If no component in an authentic-pair case clears the difference threshold, the most
changed component is assigned as target so every forged sample has target supervision.

**Advantage over the original BusterNet paper**: Wu et al. had no real-world source/target
ground truth and were forced to generate 100,000 synthetic COCO-based copy-move samples.
We derive clean source/target labels from real scientific images using the authentic-forged
pixel difference, giving us higher-quality, domain-specific training data.

## Architecture

```
Input: forged image (B, 3, 448, 448)
       ↓
DINOv2 ViT-B/14 (frozen)  ← shared; see Backbone Adaptation note below
       ↓
Features: (B, 768, 32, 32)
       ↙                         ↘
Mani-Det branch              Simi-Det branch
───────────────              ───────────────
3 conv blocks                SelfCorrelPercPooling
768→384→192→96               (B,768,32,32) → cosine similarity
→ Conv2d(96,3,1)             (B,1024,1024) → percentile pool
→ (B,3,32,32)                → (B,100,32,32)
                             3 conv blocks
                             100→128→64
                             → Conv2d(64,3,1)
                             → (B,3,32,32)
       ↘                         ↙
       Fusion: concat → (B,6,32,32)
       BN-Inception (simplified: Conv2d(6,3,1) + BN + ReLU)
       → Conv2d(3,3,3,padding=1) + softmax
       bilinear upsample → (B,3,448,448)
       3-class output: [background, target, source]
```

At inference for evaluation: `P(target) + P(source)` → forgery probability → binary mask.

### Backbone Adaptation Note

The original BusterNet uses **two separate VGG-16 feature extractors** (same architecture,
different weights). Since our DINOv2 is always frozen, one shared frozen encoder is
equivalent — both branches receive identical features, same result as two identical
frozen encoders. This is an intentional adaptation, not a loss of fidelity.

## SelfCorrelPercPooling

The key Simi-Det innovation. Computes all-pairs feature similarity across the DINOv2
spatial grid, then distils the per-location similarity distribution via percentile pooling.

### Similarity metric: cosine similarity (deviation from paper, justified)

The original paper uses **Pearson correlation** (z-score normalise each feature vector,
then compute dot products). This was appropriate for VGG-16 features, which have
unconstrained activations with varying mean and scale per location.

**We use cosine similarity (L2 normalisation) instead**, because:
- DINOv2 uses LayerNorm throughout — features are already well-scaled per location
- DINOv2 is trained with a self-distillation objective whose loss is a dot product over
  L2-normalised features. Cosine similarity is the metric the backbone was optimised for
- Every downstream use of DINOv2 in the retrieval/matching literature uses cosine similarity
- Removing the mean (z-score step) would distort DINOv2's learned semantic directions
- L2 normalisation (`F.normalize`) is simpler and better-optimised in PyTorch

**Step 1 — L2-normalise each spatial location's feature vector (GPU-native):**
```
F_flat = F.view(B, C, H*W)                    # (B, 768, 1024)
F_norm = F.normalize(F_flat, dim=1)           # (B, 768, 1024), unit vectors
```

**Step 2 — All-pairs cosine similarity (one-shot matmul, stays on GPU):**
```
S = bmm(F_norm.transpose(1, 2), F_norm)       # (B, 1024, 1024)
# S[b,i,j] = cosine similarity between location i and j, ∈ [-1, 1]
```

**Step 3 — Percentile Pooling (K=100, faithful to paper):**
For each location i, sort its 1024 similarity scores descending.
Pick K=100 scores at evenly spaced percentile positions through the sorted vector:
```
k' = round(p_k × (L - 1))   for p_k = k/(K-1), k = 0..K-1
P[b,i,k] = sorted_S[b,i,k']
```
This captures the full shape of the similarity distribution, not just the top values.
A copy-move location shows a sharp drop in similarity after the matching patch;
the percentile curve makes this pattern detectable regardless of input size.
Result: `(B, 1024, 100)` → permute → `(B, 100, 1024)` → reshape → `(B, 100, 32, 32)`

All operations GPU-native. Memory: (1024×1024) × 4B × B ≈ 16MB per batch of 4 — fine.

## Fusion Module

Paper uses `BN-Inception 3@[1,3,5]` (three parallel Conv2d branches with kernel sizes
1, 3, 5, concatenated, then BN) followed by `Conv2d(1 filter, 3×3) + softmax`.

Our simplified but faithful adaptation:
```
concat(mani_out, simi_out)  →  (B, 6, 32, 32)
Conv2d(6, 3, 1) + BN + ReLU                  ← simplified BN-Inception
Conv2d(3, 3, 3, padding=1)                   ← final classifier
softmax(dim=1)
bilinear upsample to (B, 3, 448, 448)
```

The simplification reduces the multi-kernel Inception block to a single 1×1 conv,
which is acceptable given that DINOv2 features are already richer than VGG-16.

## Training Objective and Multi-Stage Curriculum

Paper (Section 4.2, Table 1) validates that stage-wise training significantly improves
source recall (34.1%→41.6%) and target recall (47.4%→53.6%) vs. direct joint training.
Three stages, DINOv2 always frozen:

### Stage 1 — Independent branch pre-training (auxiliary tasks)

Train each branch separately with binary BCE. Separate optimizers, no cross-branch
gradients. Only cases with authentic pairs used (source/target labels available).

- **Mani-Det**: supervised on derived **target mask** — pasted region has visual artifacts.
  `L_mani = BCEWithLogitsLoss(mani_binary_logit, target_mask_float)`
- **Simi-Det**: supervised on derived **source mask** — copy origin is self-similar to target.
  `L_simi = BCEWithLogitsLoss(simi_binary_logit, source_mask_float)`

Binary logit = sigmoid of the sum of that branch's 3-channel output (channels 1+2),
keeping the auxiliary task simple. LR: `1e-2` (paper).

### Stage 2 — Freeze branches, train Fusion only

Freeze all Mani-Det and Simi-Det parameters. Train only the Fusion module.
Loss: `CrossEntropyLoss(weight=[0.1, 1.0, 1.0])` on 3-class labels.
LR: `1e-2` (paper). Initial training uses only paired cases with reliable source/target
labels.

```
labels ∈ {0=background, 1=target, 2=source}  per pixel  (B, H, W) long tensor
```

### Stage 3 — Unfreeze branches, end-to-end fine-tuning

Unfreeze Mani-Det + Simi-Det + Fusion (DINOv2 remains frozen).
Same CrossEntropyLoss. LR: `1e-5` (paper). LR reduction: halve when validation loss
plateaus, stop when no improvement for a patience window.

At inference: `forgery_prob = softmax(logits)[:, 1] + softmax(logits)[:, 2]`

## Dataset Label Format

One dataset class for all stages. Always emits `(image, label_map)` where:
- `label_map`: `(H, W)` long tensor with values `{0, 1, 2}`; batched shape is
  `(B, H, W)`, not one-hot encoded
- target/source masks are read from precomputed `data/train_masks_target/` and
  `data/train_masks_source/`
- initial BusterNet training filters forged samples to metadata
  `status == "derived_from_pair"` so target-only no-pair cases do not corrupt the
  source/target objective
- Stage 1 losses derive binary masks on the fly: `(label_map == 1).float()` for target,
  `(label_map == 2).float()` for source
- Stage 2+3: `label_map` used directly with CrossEntropyLoss

## Constraints

- DINOv2 encoder is **always frozen** (trains in minutes, same as baseline).
- Everything stays on GPU: no numpy/CPU operations during forward/backward pass.
- Input size: 448×448 (same as baseline) → 32×32 DINO feature grid (1024 locations).
- Output is always `(B, 3, H, W)` logits before softmax.
- Checkpoints and results live entirely within `einar_busternet/artifacts/`.

## Adaptations from Paper (summary)

All adaptations are documented with justification. Where the paper's design was driven
by VGG-16 constraints, we apply the modern equivalent for DINOv2.

| Aspect | Original BusterNet | Our Adaptation | Justification |
|---|---|---|---|
| Backbone | VGG-16 (pretrained ImageNet) | DINOv2 ViT-B/14 (frozen) | State-of-the-art features; frozen → no retraining; aligns with project baseline |
| Branch backbones | Two separate VGG-16s | One shared frozen DINOv2 | Frozen encoder outputs are identical for both branches — one encoder is equivalent |
| Similarity metric | Pearson correlation (z-score) | Cosine similarity (L2 norm) | DINOv2 trained with cosine-based loss; z-score distorts learned semantic directions |
| Feature grid | 16×16×512 | 32×32×768 | Follows from 448×448 input + ViT-B/14 patch stride; finer spatial resolution |
| Correlation matrix | 256×256 | 1024×1024 | Larger grid, still GPU-tractable on 4080 Super (~16MB/batch) |
| Percentile pooling | K=100 | K=100 | Faithful to paper |
| Decoder | 4-stage BN-Inception + BilinearUpPool | 3 conv blocks + bilinear upsample | VGG needed 4× upsampling stages; DINOv2 grid needs only one upsample; lighter is sufficient |
| Fusion module | BN-Inception 3@[1,3,5] + Conv2d | Conv2d(6,3,1)+BN+ReLU + Conv2d(3,3,3) | DINOv2 features are already rich; multi-scale Inception fusion is over-engineered |
| Training data | 100K synthetic COCO samples | 2377 real scientific image pairs initially; 374 no-pair cases reserved | Real domain-specific data; avoid target-only labels corrupting source learning |
| External mani data | IFS-TC + Wild Web datasets | None | Time constraint; noted as a limitation |
| Image size | 256×256 | 448×448 | Matches project baseline and pipeline |
| LR scheduling | Halve on plateau, patience=20 | ReduceLROnPlateau, tighter patience | Training runs in minutes, not days; aggressive patience is meaningless at our scale |
| Class weighting | None needed (balanced synthetic data) | `[0.1, 1.0, 1.0]` | Real data is severely imbalanced (~95% background pixels) |

## Comparison with Baseline

| | DINO Baseline | DINO BusterNet (this) |
|---|---|---|
| Backbone | DINOv2 ViT-B/14 (frozen) | DINOv2 ViT-B/14 (frozen) |
| Branches | 1 (mani-det only) | 2 (mani-det + simi-det) |
| Self-similarity | No | Yes (cosine SelfCorrelPercPooling) |
| Output channels | 1 (binary) | 3 (bg/target/source) |
| GT labels | Union mask | Derived source/target (3-class) |
| Training signal | BCE | Stage-wise BCE → CrossEntropy |

Scientific question: does explicit copy-move similarity modeling improve over
single-branch DINO segmentation for this task?
