from pathlib import Path

from pipeline import pipeline
from dataset import Datasets


def main():
    output_dir = Path(__file__).resolve().parents[1] / "artifacts" / "dino_single_stage1_run"
    pipeline(
        datasets=Datasets.ALL_TRAIN,
        image_size=3000,
        feature_backbone="dino_single",
        use_dino_transform=False,
        cnn_backbone="pretrained",
        cnn_feature_norm=True,
        pm_use_non_local=True,
        resume=True,
        batch_size=1,
        pm_iters=20,
        pm_beta=10.0,
        pm_hard_selection=True,
        pm_flat_threshold=0.15,
        pm_margin_threshold=0.10,
        pm_topk=1,
        dino_model_name="dinov2_vits14",
        dino_match_native_resolution=True,
        localization_resolution="feature_grid",
        train_feature_backbone=False,
        pm_reduced_precision=True,
        epochs=40,
        test_run=False,
        save_predictions=False,
        validation_split=0.1,
        output_dir=output_dir,
        learning_rate=1e-4, 
        mprime_loss_weight=0.6,
        empty_target_penalty_weight=0.5,
        post_process_confident_threshold=0.9,
        post_process_threshold=0.6,
        post_process_apply_closing=True,
        post_process_min_component_area=64,
    )

if __name__ == '__main__':
    main()
