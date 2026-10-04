import numpy as np
from typing import Tuple, List, Dict
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim


def dice_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    intersection = np.sum(pred * target)
    return float(2 * intersection / (np.sum(pred) + np.sum(target) + 1e-8))


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    return float(intersection / (union + 1e-8))


def precision_recall(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    tp = np.sum(pred * target)
    fp = np.sum(pred * (1 - target))
    fn = np.sum((1 - pred) * target)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return float(prec), float(rec)


def pointing_game(heatmap: np.ndarray, mask: np.ndarray, tolerance: int = 0) -> bool:
    if np.sum(mask) == 0:
        return False
    max_loc = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    if tolerance == 0:
        return bool(mask[max_loc])
    y, x = max_loc
    h, w = mask.shape
    y0 = max(0, y - tolerance)
    y1 = min(h, y + tolerance + 1)
    x0 = max(0, x - tolerance)
    x1 = min(w, x + tolerance + 1)
    return bool(np.any(mask[y0:y1, x0:x1]))


def lesion_auroc(heatmap: np.ndarray, mask: np.ndarray) -> float:
    if np.sum(mask) == 0 or np.sum(mask) == mask.size:
        return 0.5
    flat_h = heatmap.flatten()
    flat_m = mask.flatten()
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(flat_m, flat_h))


def binarize_heatmap(heatmap: np.ndarray, method: str, **kwargs) -> np.ndarray:
    if method == "threshold":
        thresh = kwargs.get("threshold", 0.5)
        return (heatmap >= thresh).astype(np.float32)
    elif method == "top_fraction":
        frac = kwargs.get("fraction", 0.1)
        n_pixels = int(heatmap.size * frac)
        thresh = np.partition(heatmap.flatten(), -n_pixels)[-n_pixels]
        return (heatmap >= thresh).astype(np.float32)
    else:
        raise ValueError(f"Unknown binarization method: {method}")


def random_baseline(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_lesion = np.sum(mask)
    if n_lesion == 0:
        return np.zeros_like(mask)
    flat = np.zeros(mask.size, dtype=np.float32)
    indices = rng.choice(mask.size, size=int(n_lesion), replace=False)
    flat[indices] = 1.0
    return flat.reshape(mask.shape)


def center_gaussian_baseline(mask: np.ndarray, sigma_frac: float = 0.2) -> np.ndarray:
    h, w = mask.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    sigma = max(h, w) * sigma_frac
    g = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return (g / (g.max() + 1e-8)).astype(np.float32)


def inverted_green_baseline(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        green = image[:, :, 1]
    else:
        green = image
    inv = 1.0 - (green / 255.0)
    return inv.astype(np.float32)


def deletion_insertion_auc(model, image: np.ndarray, heatmap: np.ndarray, target_class: int, steps: int = 20) -> Tuple[float, float]:
    h, w = heatmap.shape
    order = np.argsort(heatmap.flatten())[::-1]
    coords = np.unravel_index(order, (h, w))
    probs_del = []
    probs_ins = []
    blurred = gaussian_filter(image, sigma=10)
    for i in range(steps + 1):
        mask_del = np.ones((h, w), dtype=bool)
        mask_ins = np.zeros((h, w), dtype=bool)
        n = int(i * h * w / steps)
        mask_del[coords[0][:n], coords[1][:n]] = False
        mask_ins[coords[0][:n], coords[1][:n]] = True
        img_del = image.copy()
        img_del[mask_del == False] = blurred[mask_del == False]
        img_ins = blurred.copy()
        img_ins[mask_ins] = image[mask_ins]
        with torch.no_grad():
            logits_del = model(img_del.unsqueeze(0))
            logits_ins = model(img_ins.unsqueeze(0))
            prob_del = torch.softmax(logits_del, dim=-1)[0, target_class].item()
            prob_ins = torch.softmax(logits_ins, dim=-1)[0, target_class].item()
        probs_del.append(prob_del)
        probs_ins.append(prob_ins)
    return float(np.trapz(probs_del)), float(np.trapz(probs_ins))


import torch