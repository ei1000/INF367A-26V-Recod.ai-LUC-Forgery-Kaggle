from pipeline import pipeline
from dataset import Datasets


def main():
    pipeline(
        datasets=Datasets.TRAIN,
        feature_backbone="dino",
        test_run=True
    )


if __name__ == '__main__':
    main()