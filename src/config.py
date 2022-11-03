from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
RAW_DIR = ROOT_DIR / "raw"
IMG_DIR = RAW_DIR / "images"
IMG_SHAPE = (48, 48)
TAR_NAME = "dataset.tar.gz"
DATASET_NAME = "fer2013.csv"
