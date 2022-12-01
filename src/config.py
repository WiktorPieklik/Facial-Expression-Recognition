from pathlib import Path
import os

ROOT_DIR = Path(os.path.abspath('')).parent.parent
RAW_DIR = ROOT_DIR / "raw"
IMG_DIR = RAW_DIR / "images"
IMG_SHAPE = (48, 48)
TAR_NAME = "dataset.tar.gz"
DATASET_NAME = "fer2013.csv"
