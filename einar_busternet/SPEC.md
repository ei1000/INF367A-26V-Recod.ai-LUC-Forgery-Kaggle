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
→ (B,96,32,32)               (B,1024,1024) → percentile pool
→ aux Conv2d(96,1,3)         → (B,100,32,32)
→ target logits              3 conv blocks
                             3 conv blocks
                             100→128→64
                             → (B,64,32,32)
                             → aux Conv2d(64,1,3)
                             → copy-move union logits
       ↘                         ↙
       Fusion: concat decoder features + aux logits → (B,162,32,32)
       Conv2d(162,128,1) + BN + ReLU
       → Conv2d(128,128,3) + BN + ReLU
       → Conv2d(128,64,3) + BN + ReLU
       → Conv2d(64,out,3,padding=1)
       bilinear upsample → (B,3,448,448)
       raw 3-class logits: [background, target, source]
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

All operations GPU-native. Use a small normalisation epsilon so degenerate feature vectors
stay finite. For the initial implementation, keep the diagonal self-similarity term; it
can be removed later as an ablation if it appears to waste one percentile slot.
Memory: (1024×1024) × 4B × B ≈ 16MB per batch of 4 — fine.

## Fusion Module

Paper fuses the two branch mask-decoder feature maps. It uses `BN-Inception 3@[1,3,5]`
(three parallel Conv2d branches with kernel sizes 1, 3, 5, concatenated, then BN)
followed by `Conv2d(..., 3×3) + softmax`.

Our current competition-oriented adaptation:
```
concat(mani_features, simi_features, mani_logit, simi_logit) → (B, 162, 32, 32)
Conv2d(162, 128, 1) + BN + ReLU
Conv2d(128, 128, 3, padding=1) + BN + ReLU
Conv2d(128, 64, 3, padding=1) + BN + ReLU
Conv2d(64, out_channels, 3, padding=1)        ← final classifier
raw logits
bilinear upsample to (B, 3, 448, 448)
```

The auxiliary logits give the fusion head direct Mani target evidence and Simi union
evidence while retaining the richer decoder features. `out_channels=3` for
`three_class`; `out_channels=1` for `binary_union`.
Branch classifiers are explicit one-channel auxiliary heads:
`Conv2d(96,1,3,padding=1)` for Mani-Det and `Conv2d(64,1,3,padding=1)` for Simi-Det.
Softmax is applied by losses/evaluation, not inside the training forward pass.

## Training Objective and Multi-Stage Curriculum

Paper (Section 4.2, Table 1) validates that stage-wise training significantly improves
source recall (34.1%→41.6%) and target recall (47.4%→53.6%) vs. direct joint training.
Three stages, DINOv2 always frozen:

### Stage 1 — Independent branch pre-training (auxiliary tasks)

Train each branch separately with BCE+soft-Dice. Separate optimizers, no cross-branch
gradients. Initial training uses clean paired cases plus their authentic counterparts.
Forged samples use only `status == "derived_from_pair"` source/target labels; authentic
samples are all-background labels so the model also learns to suppress false positives,
which matters for the official oF1 score.

- **Mani-Det**: supervised on derived **target mask** — pasted region has visual artifacts.
  `L_mani = BCEWithLogitsLoss + branch_dice_weight * SoftDiceLoss`
- **Simi-Det**: supervised on derived **source+target union mask** — self-similarity is
  symmetric and should detect both duplicated regions, not decide which one was pasted.
  `L_simi = BCEWithLogitsLoss + branch_dice_weight * SoftDiceLoss`

Use raw one-channel auxiliary branch logits:
`mani_logits[:, 0]` for target supervision and `simi_logits[:, 0]` for source+target
union supervision. Do not apply sigmoid before BCE/Dice; Dice applies sigmoid internally.
Current LR: `1e-3`.

### Stage 2 — Freeze branches, train Fusion only

Freeze all Mani-Det and Simi-Det parameters. Train only the Fusion module.
Default loss: `CrossEntropyLoss(weight=[0.3, 1.0, 1.0])` on 3-class labels.
Binary-fusion ablation: BCE+soft-Dice on `(label_map > 0).float()`.
LR: `1e-2` (paper). Initial training uses only paired forged cases with reliable
source/target labels plus their authentic counterparts as all-background negatives.

```
labels ∈ {0=background, 1=target, 2=source}  per pixel  (B, H, W) long tensor
```

### Stage 3 — Unfreeze branches, end-to-end fine-tuning

Unfreeze Mani-Det + Simi-Det + Fusion (DINOv2 remains frozen).
Same fusion loss as Stage 2. LR: `1e-5` (paper). LR reduction: halve when validation loss
plateaus, stop when no improvement for a patience window.

At inference/evaluation, wrap the model as a binary foreground model. For the 3-class
model: `forgery_prob = softmax(logits)[:, 1] + softmax(logits)[:, 2]`. For the binary
fusion ablation, the wrapper passes the one-channel logits through. This lets us reuse
the baseline validation and oF1 scoring path.

Conceptually:

```python
class BusterNetUnionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        if logits.shape[1] == 1:
            return logits
        probs = logits.softmax(dim=1)
        forgery_prob = probs[:, 1:2] + probs[:, 2:3]
        return torch.logit(forgery_prob.clamp(1e-6, 1 - 1e-6))
```

The wrapper returns one-channel binary logits because the existing baseline evaluator
applies `sigmoid` internally.

## Dataset Label Format

One dataset class for all stages. It follows the baseline `ForgeryDataset` preprocessing
style: RGB images by default, optional DINO/ImageNet normalisation, bilinear image resize,
and nearest-neighbor label resize. Always emits `(image, label_map)` where:
- `label_map`: `(H, W)` long tensor with values `{0, 1, 2}`; batched shape is
  `(B, H, W)`, not one-hot encoded
- target/source masks are read from precomputed `data/train_masks_target/` and
  `data/train_masks_source/`
- initial BusterNet training keeps forged samples only when metadata has
  `status == "derived_from_pair"` so target-only no-pair cases do not corrupt the
  source/target objective
- authentic samples are included as all-background labels, but for the initial run they
  are limited to case IDs that have a paired `derived_from_pair` forged sample
- forged label construction follows the class convention:
  `label_map[target_mask > 0] = 1`, then `label_map[source_mask > 0] = 2`
- Stage 1 losses derive binary masks on the fly: `(label_map == 1).float()` for Mani
  target supervision, `(label_map > 0).float()` for Simi union supervision
- Stage 2+3: `label_map` used directly with CrossEntropyLoss for 3-class fusion, or
  converted to `(label_map > 0).float()` for binary fusion

## Constraints

- DINOv2 encoder is **always frozen** (trains in minutes, same as baseline).
- Everything stays on GPU: no numpy/CPU operations during forward/backward pass.
- Input size: 448×448 (same as baseline) → 32×32 DINO feature grid (1024 locations).
- Default output is `(B, 3, H, W)` logits before softmax. The binary fusion ablation
  outputs `(B, 1, H, W)` raw binary logits.
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
| Fusion module | Decoder-feature fusion with BN-Inception 3@[1,3,5] + Conv2d | Decoder features plus auxiliary logits, Conv2d(162,128,1) + two 3x3 blocks + classifier | DINO dominates runtime, so a wider fusion head is cheap; auxiliary logits provide direct Mani/Simi evidence |
| Training data | 100K synthetic COCO samples | 2377 real scientific image pairs initially; 374 no-pair cases reserved | Real domain-specific data; avoid target-only labels corrupting source learning |
| External mani data | IFS-TC + Wild Web datasets | None | Time constraint; noted as a limitation |
| Image size | 256×256 | 448×448 | Matches project baseline and pipeline |
| LR scheduling | Halve on plateau, patience=20 | ReduceLROnPlateau, tighter patience | Training runs in minutes, not days; aggressive patience is meaningless at our scale |
| Class weighting | None needed (balanced synthetic data) | `[0.3, 1.0, 1.0]` or binary union BCE | Real data is severely imbalanced; competition scores union masks |

## Comparison with Baseline

| | DINO Baseline | DINO BusterNet (this) |
|---|---|---|
| Backbone | DINOv2 ViT-B/14 (frozen) | DINOv2 ViT-B/14 (frozen) |
| Branches | 1 (mani-det only) | 2 (mani-det + simi-det) |
| Self-similarity | No | Yes (cosine SelfCorrelPercPooling) |
| Output channels | 1 (binary) | 3 (bg/target/source) or binary union ablation |
| GT labels | Union mask | Derived source/target (3-class) |
| Training signal | BCE | Stage-wise BCE → CE or binary union BCE |

Scientific question: does explicit copy-move similarity modeling improve over
single-branch DINO segmentation for this task?
