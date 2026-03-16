from pipeline import pipeline
from dataset import Datasets


def main():
    pipeline(Datasets.SUPPLEMENT, test_run=True)


if __name__ == '__main__':
    main()