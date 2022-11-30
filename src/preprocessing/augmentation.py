import cv2
import imgaug.augmenters as iaa
import os
from random import sample
from typing import List, Dict

import numpy as np


def save_images(dataset_path: str, augmented_images: Dict[str, List[np.ndarray]]) -> None:
    counter = 1
    for emotion, imgs in augmented_images.items():
        for img in imgs:
            path = f"{dataset_path}/{emotion}/aug_{counter:04d}.jpg"
            cv2.imwrite(path, img)
            counter += 1


class BalancedAugmenter:
    def __init__(self):
        self.__sequence = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Crop(percent=(0, 0.1)),
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            iaa.LinearContrast((.75, 1.5)),
            iaa.Multiply((0.8, 1.2), per_channel=False),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=True)

    @staticmethod
    def calculate_classes_distribution(dataset_path: str) -> dict:
        distribution = {}
        for root, _, files in os.walk(dataset_path):
            if root == dataset_path:
                continue
            else:
                emotion = os.path.basename(root)
                distribution[emotion] = len(files)

        return distribution

    def apply(self, dataset_path: str, save: bool = False) -> Dict[str, List[np.ndarray]]:
        distribution = self.calculate_classes_distribution(dataset_path)
        biggest_class = max(distribution, key=lambda k: distribution[k])
        augmented_imgs = {}
        for root, _, files in os.walk(dataset_path):
            if root == dataset_path:
                continue
            else:
                img_count = len(files)
                imgs_to_augment = distribution[biggest_class] - img_count
                if imgs_to_augment:
                    emotion = os.path.basename(root)
                    augmented_imgs[emotion] = self.__apply(root, imgs_to_augment, img_count)

        if save:
            save_images(dataset_path, augmented_imgs)

        return augmented_imgs

    def __apply(self, path_to_imgs: str, imgs_to_augment: int, img_total_count: int) -> List[np.ndarray]:
        if img_total_count - imgs_to_augment < 0:
            left_to_generate = imgs_to_augment
            img_names = []
            while left_to_generate > 0:
                batch_size = img_total_count \
                    if left_to_generate - img_total_count > 0 \
                    else left_to_generate
                img_names.extend(sample(os.listdir(path_to_imgs), k=batch_size))
                left_to_generate -= batch_size
        else:
            img_names = sample(os.listdir(path_to_imgs), k=imgs_to_augment)
        imgs = list(map(lambda img_path: cv2.imread(path_to_imgs + "/" + img_path, cv2.IMREAD_GRAYSCALE), img_names))
        augmented_imgs = self.__sequence(images=imgs)

        return augmented_imgs
