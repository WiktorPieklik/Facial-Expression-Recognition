from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
RAW_DIR = ROOT_DIR / "raw"
IMAGES_DIR = str((RAW_DIR / "images").resolve())
