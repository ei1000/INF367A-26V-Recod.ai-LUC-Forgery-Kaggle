# README.md

# Deep PatchMatch for Scientific Image Forgery Detection

## Project Overview

This project implements an adaptation of Deep PatchMatch based on  
“Image Copy-Move Forgery Detection via Deep PatchMatch and Pairwise Ranking Learning” by Li et al. (available at https://arxiv.org/pdf/2404.17310)

The method is adapted for the Kaggle competition  
**Recod.ai/LUC - Scientific Image Forgery Detection**, which differs from the original paper in both task formulation and dataset characteristics.

For the full technical report: See [`docs/technical_report.md`](docs/technical_report.md)

---

## Novel Method Description

This project adapts Deep PatchMatch from a **three-class CMFD task** (background, source, target) to a **binary segmentation task** (authentic vs forged pixels).

Key modifications include:

- Use of pretrained frozen CNNs (VGG16 / ResNet18) instead of end-to-end training
- Integration of DINOv2 features in a parallel SEUNet branch
- Modified Zernike polynomial formulation
- Added offset re-randomization in PatchMatch to avoid local optima
- Incorporation of Zernike offsets in the DLF stage
- Use of post-processing techniques to improve prediction quality

---

## Implementation

The method is implemented across modular components:

- Feature extraction (`feature_extractors/`)
- PatchMatch (`cross_scale_patchmatch/`)
- Prediction + DLF (`prediction/`)
- Full pipeline (`pipeline.py`)

The implementation follows the structure of the original architecture, but is not fully end-to-end differentiable due to computational and practical constraints.

---

## Model Evaluation

The model is evaluated on the competition dataset using the OF1 metric, which strongly penalizes false positives and fragmented predictions.

Main observations:

- High recall but low precision
- Strong sensitivity to post-processing
- Performance impacted by differences between natural and scientific images

More detailed analysis is provided in the technical report.

## TODO: ADD SOME EVAL METRICS

## Notes

- Some implementation details were not specified in the original paper and required independent design choices
- LLMs were used for optimization and implementation support in selected components

See the technical report for full details.
