# Introduction

This project uses uv, can be installed with;

- `pip install uv`

Then env is initiated with;

- `uv venv`

The env is maintained with;

- `uv sync` - If new dependencies are not downloaded locally
- `uv lock` - If you add new dependencies

The main entrypoint of the project is the `train_baseline.py` file. This runs a pipeline consisting of the DINOv2 feature extractor, a simple decoder and some pixel mask post-processing operations.

## Report

Our methods are documented in detail in `report/nldl-NLDL2025/main.pdf`, where we give an overview of the different models utilized and more information on the individual competition.

See also the respective READMEs in the `x_individual_project` directories for more information on the specific implementations.

## Kaggle Competition

The aim of this project is to achieve good performance on the Kaggle competition: "Recod.ai/LUC - Scientific Image Forgery Detection". This competition aims to predict Copy-Move Forgeries on scientific images. See the competition here: https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection. Our implementation utilizes the training data and metric from the competition. Unfortunately, we were unable to enter the competition, so we were neither able to use the associated test data nor submit our model to the Kaggle leaderboard. For more detailed information on the competition, see the report.

## Data

- Data is stored in data/ folder. It must be added it if not present in order to run the project.

## Individual contributions

We
