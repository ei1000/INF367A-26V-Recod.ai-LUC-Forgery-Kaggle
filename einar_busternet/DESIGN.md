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
progressive decoder      SelfCorrelPercPooling → (B, 100, 32, 32)
32→64→128 spatial        progressive decoder, 32→64→128 spatial
768→512→256→192→128      100→256→192→128→96
aux Conv2d(128,1,3)      aux Conv2d(96,1,3)
(B, 128, 128, 128)       (B, 96, 128, 128)
   ↘                      ↙
concat decoder features + aux logits → (B, 226, 128, 128)
Fusion: parallel Conv2d(226,64,k={1,3,5}) → concat 192 + BN/ReLU
        → Conv2d(192,128,3)+BN+ReLU → Conv2d(128,64,3)+BN+ReLU
        → Conv2d(64,out,3,pad=1)
bilinear upsample → (B, out, 448, 448)
out=1 for binary union, out=3 for source/target experiment
```

Main inference path: binary union logits. For the 3-class experiment,
`forgery_prob = P(target) + P(source)`.

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
- 2377/2751 forged cases have authentic pairs; 374 no-pair cases are reserved for later

**Advantage over paper**: Wu et al. required 100K synthetic COCO samples. We initially
use 2377 real scientific image pairs with domain-specific source/target labels.

## Training Stages

| Stage | Trains | Loss | LR | Labels |
|---|---|---|---|---|
| 1a — Mani-Det | mani decoder + classifier | BCE+Dice | 1e-3 | target mask |
| 1b — Simi-Det | simi decoder + classifier + corr | BCE+Dice | 1e-3 | source+target union mask |
| 2 — Fusion | fusion only | binary BCE+Dice | 1e-2 | source+target union |
| 3 — Fine-tune | branches + fusion (DINOv2 frozen) | binary BCE+Dice + 0.1 aux losses | 1e-5 | union + branch auxiliaries |

Class weights `[0.3, 1.0, 1.0]` are the current 3-class setting after validation showed
that `[0.1, 1.0, 1.0]` caused too many authentic false positives. The submitted model
uses `fusion_mode="binary_union"` because it better matches the scored union-mask task.

## Key Deviations from Paper

| Paper | Ours | Why |
|---|---|---|
| VGG-16, two separate extractors | DINOv2, one shared frozen | Better features; frozen = identical outputs |
| Pearson correlation (z-score) | Cosine similarity (L2) | DINOv2 optimised for cosine |
| 4-stage BN-Inception decoder | Progressive 32→64→128 branch decoders | Closer to BusterNet and modern dense prediction; helps small/source regions before final upsample |
| Multi-kernel Inception fusion | Decoder features + aux logits, parallel 1×1/3×3/5×5 fusion → classifier | Closer to BusterNet fusion; aux logits expose direct Mani/Simi evidence |
| 100K synthetic samples | 2377 real pairs initially | Real > synthetic for domain-specific task |
| No class weighting | Binary union BCE+Dice main path | Real data is imbalanced; Kaggle scores union masks |
