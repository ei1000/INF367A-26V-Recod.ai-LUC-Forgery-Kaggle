from pipeline import pipeline
from dataset import Datasets


def main():
    pipeline(
        datasets=Datasets.SELF_PROCURED,
        feature_backbone="cnn",
        cnn_backbone="pretrained",
        cnn_pretrained_model="resnet18",
        cnn_feature_norm=True,  
        test_run=True,
    )

if __name__ == '__main__':
    main()