from pipeline import pipeline
from dataset import Datasets


def main():
    pipeline(
        datasets=Datasets.SUPPLEMENT,
        feature_backbone="cnn",
        cnn_backbone="pretrained",
        cnn_pretrained_model="vgg16_bn", 
        test_run=True,
    )

if __name__ == '__main__':
    main()