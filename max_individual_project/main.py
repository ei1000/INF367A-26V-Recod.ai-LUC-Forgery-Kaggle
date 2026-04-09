from pathlib import Path

from pipeline import pipeline
from dataset import Datasets


def main():
    output_dir = Path(__file__).resolve().parents[1] / "artifacts" / "cnn_pretrained_frozen_run"
    pipeline(
        datasets=Datasets.ALL_TRAIN,
        image_size=448,
        feature_backbone="cnn",
        use_dino_transform=False,
        cnn_backbone="pretrained",
        cnn_feature_norm=True,
        pm_use_non_local=True,
        resume=True,
        batch_size=4,
        pm_iters=24,
        pm_beta=10.0,
        pm_hard_selection=True,
        dino_match_native_resolution=False,
        train_feature_backbone=False,
        pm_reduced_precision=True,
        epochs=40,
        test_run=False,
        save_predictions=False,
        validation_split=0.1,
        output_dir=output_dir,
        learning_rate=1e-4,
        mprime_loss_weight=0.6,
    )

if __name__ == '__main__':
    main()
