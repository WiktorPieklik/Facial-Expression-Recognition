from cryptography.fernet import Fernet
from typing import List
from decouple import config, UndefinedValueError
from pandas import read_csv
from matplotlib.pyplot import imsave
from alive_progress import alive_bar
import numpy as np
from simple_chalk import chalk
from src.config import RAW_DIR, IMG_DIR, IMG_SHAPE, TAR_NAME, DATASET_NAME
from io import BytesIO
import os
import tarfile


def info(msg: str) -> None:
    print(chalk.yellow(msg))


def success(msg: str) -> None:
    print(chalk.green(msg))


def error(msg: str) -> None:
    print(chalk.red(msg))


def is_dataset_empty() -> bool:
    dataset_path = str(IMG_DIR.resolve())
    is_empty = True
    directory_content = []
    try:
        directory_content = os.listdir(dataset_path)
    except FileNotFoundError:
        pass

    if len(directory_content) != 0:
        is_empty = False

    return is_empty


def emotion_name(emotion_no: int) -> str:
    return {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }[emotion_no]


def decrypt_file(file_path: str) -> bytes:
    file = None
    with open(file_path, 'rb') as file_handler:
        file = file_handler.read()
    try:
        key = config('TAR_KEY')
    except UndefinedValueError:
        error("You haven't provided tar key in your .env file!")
        return
    fernet = Fernet(key)

    return fernet.decrypt(file)


def untar_archive(archive_file: bytes, destination: str) -> List[str]:
    archive = BytesIO(archive_file)
    files_to_delete = []
    with tarfile.open(fileobj=archive) as tar:
        tar.extractall(destination)
        for member in tar.getmembers():
            files_to_delete.append(member.path)

    return files_to_delete


def save_img(img: np.array, emotion_no: int, usage: str, i: int) -> None:
    img_name = f"{i:04}.jpg"
    destination = IMG_DIR / usage / emotion_name(emotion_no)
    os.makedirs(str(destination.resolve()), exist_ok=True)
    imsave(str((destination / img_name).resolve()), img, cmap='gray')


if __name__ == '__main__':
    if is_dataset_empty():
        tar_path = str((RAW_DIR / TAR_NAME).resolve())
        destination = str(IMG_DIR.resolve())
        info("Decrypting dataset")
        encrypted_tar = decrypt_file(tar_path)
        relative_paths_to_be_deleted = untar_archive(encrypted_tar, destination)
        dataset_path = str((IMG_DIR / DATASET_NAME).resolve())
        info("Reading csv file, this may take a while...")
        dataset = read_csv(dataset_path)
        count = int(dataset.count()[0])
        with alive_bar(count, dual_line=True, title="Creating images from csv") as bar:
            for i, row in dataset.iterrows():
                # extract image
                emotion_no = row['emotion']
                usage = row['Usage']
                bar.text(f"-> Directory {usage}, Emotion: {emotion_name(emotion_no)}")
                image = np.array(row["pixels"].split()).astype(np.uint)
                image = image.reshape(IMG_SHAPE)
                save_img(image, emotion_no, usage, i + 1)
                bar()
        info("Cleaning dataset directory")
        for relative_path in relative_paths_to_be_deleted:
            path = str((IMG_DIR / relative_path).resolve())
            os.remove(path)
        success("Everything's done!")
    else:
        info("You have already dataset")

