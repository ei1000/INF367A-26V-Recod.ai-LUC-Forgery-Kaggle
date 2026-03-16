from pipeline import pipeline
from dataset import Datasets


def main():
    pipeline(Datasets.SUPPLEMENT, feature_backbone='dino', use_dino_transform=True, test_run=True)


if __name__ == '__main__':
    main()