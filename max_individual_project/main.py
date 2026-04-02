from pipeline import pipeline
from dataset import Datasets


def main():
    pipeline(
        datasets=Datasets.ALL_TRAIN,
        feature_backbone="cnn",
        cnn_backbone="simple",
        cnn_feature_norm=True,
        pm_use_non_local=True,
        resume=True,
        batch_size=6,
        pm_iters=8,
        epochs=50,
        test_run=False,
        save_predictions=False,
        validation_split=0.1,
        output_dir="artifacts/easter_run",
    )

if __name__ == '__main__':
    main()
