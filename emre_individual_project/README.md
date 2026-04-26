# Emre Balci - Individual Work

This folder contains the notebook for my individual contribution to the project on scientific image forgery detection.

## Main contribution

My individual work focused on designing, implementing, and evaluating a **SegNeXt-inspired segmentation baseline** and comparing it against the project's stronger **DINO-based baseline** under a shared validation protocol.

The main goal of this work was to test whether a lightweight learned segmentation approach could become competitive with the DINO pipeline when trained and evaluated under the same data regime.

## Notebook contents

The notebook documents an end-to-end experimental workflow including:

- dataset download and preparation
- subset extraction for controlled experiments
- generation of helper scripts and configuration files
- implementation of a SegNeXt-inspired segmentation model
- configuration and execution of repeated training runs
- comparison between:
  - DINO baseline
  - custom SegNeXt-inspired model
  - pretrained SegNeXt-family variant with a ConvNeXt-Tiny encoder
- evaluation across multiple training subset sizes and learning rates
- aggregation of experiment outputs from JSON summaries
- identification of best-performing runs
- inspection of validation F1 curves and loss curves

## Experimental scope

The notebook includes experiments over:

- training subset sizes: 200, 500, and 1000
- validation subset sizes matched to each setup
- learning rates: 1e-4, 5e-5, 3e-5, and 1e-5
- custom and pretrained encoder settings
- DINO and SegNeXt-based model variants

## Technical focus

The technical emphasis of this notebook is on:

1. **representation quality vs architecture choice**  
   Testing whether a SegNeXt-style architecture can close the gap to the DINO baseline.

2. **effect of pretraining**  
   Comparing custom from-scratch SegNeXt-inspired models against pretrained ConvNeXt-Tiny based variants.

3. **training stability and hyperparameter sensitivity**  
   Evaluating how learning rate, subset size, and training duration affect convergence and validation F1.

4. **reproducible experiment orchestration**  
   Generating scripts/configs programmatically and collecting outputs from saved run directories.

## Connection to the report

This notebook supports the individual sections of the report related to:

- SegNeXt-inspired baseline and experimental comparison
- validation F1 comparison against the DINO-based baseline
- transfer learning effects
- hyperparameter sensitivity
- training stability and computational trade-offs

## File

- `Emre Individual Work.ipynb` — main notebook containing the experimental workflow and analysis for my individual contribution.
