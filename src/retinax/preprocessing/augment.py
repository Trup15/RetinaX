import cv2
import numpy as np
import random


def augment_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    if rng.random() < 0.5:
        image = cv2.flip(image, 1)
    if rng.random() < 0.5:
        image = cv2.flip(image, 0)
    angle = rng.uniform(-20, 20)
    scale = rng.uniform(0.92, 1.08)
    shift_x = rng.uniform(-0.05, 0.05) * w
    shift_y = rng.uniform(-0.05, 0.05) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += shift_x
    M[1, 2] += shift_y
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = rng.uniform(0.8, 1.2)
    beta = rng.uniform(-20, 20)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    gamma = rng.uniform(0.8, 1.2)
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
    image = cv2.LUT(image, table)
    return image


def get_train_augmentation(rng: np.random.Generator):
    def _augment(image: np.ndarray) -> np.ndarray:
        return augment_image(image, rng)
    return _augment