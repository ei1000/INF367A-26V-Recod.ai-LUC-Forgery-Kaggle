from pipeline import pipeline
from dataset import Datasets


def main():
    pipeline(
        datasets=Datasets.ALL_TRAIN,
        feature_backbone="cnn",
        cnn_backbone="simple",
        cnn_feature_norm=True,  
        pm_use_non_local=True,
        batch_size=8,
        epochs=5,
        test_run=False,
    )

if __name__ == '__main__':
    main()