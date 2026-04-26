# Recod.ai/LUC - Scientific Image Forgery Detection

Group project for INF367A. The goal is copy-move forgery detection (CMFD) on scientific images
from the [Kaggle competition](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection).

## Competition

Copy-move forgery duplicates a region of an image and pastes it elsewhere, often with
post-processing to hide the manipulation. The competition focuses on scientific images, where
this technique can be used to fabricate results.

The dataset contains 2751 forged and 2377 authentic images. For 2377 forged cases an authentic
counterpart is also provided. Images vary widely in size and are resized to 448x448 for training.

The evaluation metric is a component-level oF1: pixelwise F1 is computed per connected component
and matched to ground truth using the Hungarian algorithm. A penalty is applied for extra predicted
components. Any prediction on an authentic image gives oF1=0 for that image.

We were unable to submit to the Kaggle test set due to late entry. All results are on a local
stratified holdout split (80/10/10 train/val/test, seed=42).

## Setup

This project uses [uv](https://github.com/astral-sh/uv):

```bash
pip install uv
uv venv
uv sync
```

## Data

Place the Kaggle competition data in `data/`. The expected structure is documented in `data/README.md`.

## Baseline

The baseline uses a frozen DINOv2 ViT-B/14 encoder with a lightweight convolutional decoder.
Configuration is in `configs/baseline_config.py`.

```bash
# Train
uv run python train_baseline.py

# Evaluate on holdout (one-shot, keep locked during model selection)
uv run python evaluate_baseline.py --checkpoint runs/checkpoints/best_by_kaggle_score.pt --confirm-local-holdout --allow-torch-hub
```

Checkpoints are saved to `runs/checkpoints/`. Validation diagnostics can be explored in
`notebooks/evaluate_validation_postprocess.ipynb`.

## Data Exploration

`notebooks/explore_forgery_data.ipynb` covers image dimensions, mask statistics, instance counts,
and authentic/forged pair inspection.

## Individual Projects

Each group member implemented a novel method:

- `max_crosscale_patchmatch/` — Deep Cross-Scale PatchMatch adaptation (Max)
- `einar_busternet/` — BusterNet-inspired dual-branch model with DINOv2 features (Einar)
- `emre_segnext/` — SegNeXt-inspired segmentation baseline (Emre)

See the individual READMEs for method descriptions, running instructions, and results.

## Report

Full description of methods, results, and discussion: `report/nldl-NLDL2025/main.pdf`.
