import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PreprocessParams:
    box_x0: int
    box_y0: int
    box_x1: int
    box_y1: int
    pad_top: int
    pad_left: int
    side: int
    size: int


def fov_box(image: np.ndarray) -> Tuple[int, int, int, int]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = gray.shape
        return 0, 0, w, h
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, x + w, y + h


def preprocess(
    image: np.ndarray,
    img_size: int = 384,
    clahe: bool = False,
) -> Tuple[np.ndarray, PreprocessParams]:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = fov_box(image)
    if x1 <= x0 or y1 <= y0:
        x0, y0, x1, y1 = 0, 0, w, h
    crop = image[y0:y1, x0:x1]
    side = max(crop.shape[0], crop.shape[1])
    pad_top = (side - crop.shape[0]) // 2
    pad_left = (side - crop.shape[1]) // 2
    pad_bottom = side - crop.shape[0] - pad_top
    pad_right = side - crop.shape[1] - pad_left
    padded = cv2.copyMakeBorder(
        crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
    )
    resized = cv2.resize(padded, (img_size, img_size), interpolation=cv2.INTER_AREA)
    if clahe:
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe_obj.apply(l)
        lab = cv2.merge((l, a, b))
        resized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    params = PreprocessParams(
        box_x0=x0,
        box_y0=y0,
        box_x1=x1,
        box_y1=y1,
        pad_top=pad_top,
        pad_left=pad_left,
        side=side,
        size=img_size,
    )
    return resized, params


def apply_params_to_mask(mask: np.ndarray, params: PreprocessParams, img_size: int) -> np.ndarray:
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = params.box_x0, params.box_y0, params.box_x1, params.box_y1
    if x1 <= x0 or y1 <= y0:
        x0, y0, x1, y1 = 0, 0, w, h
    crop = mask[y0:y1, x0:x1]
    side = max(crop.shape[0], crop.shape[1])
    pad_top = (side - crop.shape[0]) // 2
    pad_left = (side - crop.shape[1]) // 2
    pad_bottom = side - crop.shape[0] - pad_top
    pad_right = side - crop.shape[1] - pad_left
    padded = cv2.copyMakeBorder(
        crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
    )
    resized = cv2.resize(padded, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return resized