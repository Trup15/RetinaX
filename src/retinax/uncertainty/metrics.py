import numpy as np
from scipy.special import softmax, logsumexp
from typing import Tuple, Optional


def temperature_scale(logits: np.ndarray, T: float) -> np.ndarray:
    return softmax(logits / T, axis=-1)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray, max_iter: int = 100, lr: float = 0.01) -> float:
    T = 1.0
    for _ in range(max_iter):
        probs = temperature_scale(val_logits, T)
        grad = np.mean(probs - np.eye(probs.shape[1])[val_labels] * probs, axis=0).sum() / (T ** 2)
        if abs(grad) < 1e-6:
            break
        T -= lr * grad
        T = max(T, 1e-3)
    return float(T)


def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if i == n_bins - 1:
            mask = mask | (confidences == 1.0)
        if np.any(mask):
            bin_acc = np.mean(accuracies[mask])
            bin_conf = np.mean(confidences[mask])
            ece += np.mean(mask) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    n_classes = probs.shape[1]
    one_hot = np.eye(n_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def nll_score(probs: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-15
    probs_clipped = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(np.log(probs_clipped[np.arange(len(labels)), labels])))


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-15
    probs_clipped = np.clip(probs, eps, 1 - eps)
    entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)
    return entropy / np.log(probs.shape[1]) if probs.shape[1] > 1 else np.zeros(probs.shape[0])


def mutual_information(mc_probs: np.ndarray) -> np.ndarray:
    mean_probs = np.mean(mc_probs, axis=0)
    entropy_mean = -np.sum(mean_probs * np.log(np.clip(mean_probs, 1e-15, 1)), axis=1)
    mean_entropy = np.mean(-np.sum(mc_probs * np.log(np.clip(mc_probs, 1e-15, 1)), axis=2), axis=0)
    return entropy_mean - mean_entropy


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, ci: float = 0.95, rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng()
    n = len(values)
    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        stats.append(np.mean(values[idx]))
    alpha = (1 - ci) / 2
    return float(np.percentile(stats, alpha * 100)), float(np.percentile(stats, (1 - alpha) * 100))