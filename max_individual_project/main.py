from pipeline import pipeline
from dataset import Datasets


def main():
    pipeline(
        datasets=Datasets.SUPPLEMENT,
        feature_backbone="dino",
        dino_model_name="dinov2_vits14",
        use_dino_transform=True,
        test_run=True
    )


if __name__ == '__main__':
    main()