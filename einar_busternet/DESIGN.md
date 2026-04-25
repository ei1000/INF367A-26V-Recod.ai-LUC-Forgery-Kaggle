# BusterNet-DINO Design

BusterNet (Wu et al., ECCV 2018) adapted for DINOv2 features and scientific CMFD.

## Architecture

```
Input (B, 3, 448, 448)
    ↓
DINOv2 ViT-B/14 — frozen, shared          [modern: stronger than VGG-16; frozen = no retraining]
    ↓
(B, 768, 32, 32)
   ↙                      ↘
Mani-Det                 Simi-Det
3 conv blocks            SelfCorrelPercPooling → (B, 100, 32, 32)
768→384→192→96           3 conv blocks: 100→128→64
→ Conv2d(96, 3, 1)       → Conv2d(64, 3, 1)
(B, 3, 32, 32)           (B, 3, 32, 32)
   ↘                      ↙
concat → (B, 6, 32, 32)
Fusion: Conv2d(6,3,1)+BN+ReLU → Conv2d(3,3,3,pad=1)   [simplified BN-Inception]
bilinear upsample → (B, 3, 448, 448)
softmax → [background, target, source]
```

Inference: `forgery_prob = P(target) + P(source)`

## SelfCorrelPercPooling

```
F_flat = F.view(B, 768, 1024)
F_norm = F.normalize(F_flat, dim=1)          # cosine, not Pearson
S = bmm(F_norm.T, F_norm)                   # (B, 1024, 1024)
S_sorted = S.sort(dim=-1, descending=True)  # per-location
P = S_sorted[:, :, percentile_indices]      # K=100 evenly spaced positions
→ (B, 100, 32, 32)                          # captures full similarity distribution
```

**Cosine vs Pearson**: DINOv2 is trained with a cosine-based loss and uses LayerNorm
throughout — cosine similarity is the correct metric. Pearson (z-score) was needed for
unconstrained VGG-16 activations, not for ViT features.

## Source/Target Labels

Derived from authentic-forged pixel difference — no synthetic data needed:
- **Target**: component with ≥25% pixels changed by >5 intensity units (pasted region)
- **Source**: remaining components (copy origin, unchanged in authentic)
- ~85% of forged cases have authentic pairs; remainder falls back to target-only

**Advantage over paper**: Wu et al. required 100K synthetic COCO samples. We use 2377
real scientific image pairs with domain-specific labels.

## Training Stages

| Stage | Trains | Loss | LR | Labels |
|---|---|---|---|---|
| 1a — Mani-Det | mani decoder | BCE | 1e-2 | target mask |
| 1b — Simi-Det | simi decoder + corr | BCE | 1e-2 | source mask |
| 2 — Fusion | fusion only | CrossEntropy `[0.1,1,1]` | 1e-2 | 3-class |
| 3 — Fine-tune | all (DINOv2 frozen) | CrossEntropy `[0.1,1,1]` | 1e-5 | 3-class |

Class weights `[0.1, 1.0, 1.0]` compensate for ~95% background pixels (paper used
balanced synthetic data and needed no weighting).

## Key Deviations from Paper

| Paper | Ours | Why |
|---|---|---|
| VGG-16, two separate extractors | DINOv2, one shared frozen | Better features; frozen = identical outputs |
| Pearson correlation (z-score) | Cosine similarity (L2) | DINOv2 optimised for cosine |
| 4-stage BN-Inception decoder | 3 conv blocks + upsample | 16× vs ~14× upsampling; DINOv2 features are richer |
| Multi-kernel Inception fusion | Conv2d(6→3) + Conv2d(3→3) | Unnecessary at DINOv2 feature quality |
| 100K synthetic samples | 2377 real pairs | Real > synthetic for domain-specific task |
| No class weighting | `[0.1, 1.0, 1.0]` | Real data is imbalanced |
