from pipeline import pipeline
from dataset import Datasets


def main():
    pipeline(
        datasets=Datasets.TRAIN,
        feature_backbone="cnn",
        cnn_backbone="simple",
        cnn_feature_norm=True,  
        pm_use_non_local=True,
        test_run=True,
    )

if __name__ == '__main__':
    main()