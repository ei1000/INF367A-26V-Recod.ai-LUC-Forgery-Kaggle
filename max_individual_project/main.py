from pathlib import Path

from pipeline import pipeline
from dataset import Datasets


def main():
    output_dir = Path(__file__).resolve().parents[1] / "artifacts" / "patchmatch_improved_run"
    pipeline(
        datasets=Datasets.ALL_TRAIN,
        feature_backbone="dino",
        use_dino_transform=True,
        dino_model_name="dinov2_vitb14",
        dino_proj_dim=None,
        cnn_feature_norm=True,
        pm_use_non_local=True,
        resume=True,
        batch_size=3,
        pm_iters=8,
        pm_beta=10,
        dino_match_native_resolution=True,
        epochs=20,
        test_run=False,
        save_predictions=False,
        validation_split=0.1,
        output_dir=output_dir,
        learning_rate=1e-3,
        mprime_loss_weight=0.5,
    )

if __name__ == '__main__':
    main()
