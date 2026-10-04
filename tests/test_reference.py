import pytest
import numpy as np
import cv2
import torch
from pathlib import Path
import tempfile
import shutil

from retinax.preprocessing.preprocess import preprocess, apply_params_to_mask, PreprocessParams
from retinax.preprocessing.augment import augment_image
from retinax.uncertainty.metrics import (
    temperature_scale, fit_temperature, ece_score, brier_score, nll_score,
    predictive_entropy, mutual_information, bootstrap_ci
)
from retinax.xai.lesion_metrics import (
    dice_coefficient, iou_score, precision_recall, pointing_game,
    lesion_auroc, binarize_heatmap, random_baseline, center_gaussian_baseline
)
from retinax.config import load_config, Config
from retinax.utils import seed_everything, get_device, run_info


class TestPreprocessing:
    def test_preprocess_shape(self):
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        out, params = preprocess(img, img_size=384, clahe=False)
        assert out.shape == (384, 384, 3)
        assert isinstance(params, PreprocessParams)
        assert params.size == 384

    def test_preprocess_clahe(self):
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        out, _ = preprocess(img, img_size=224, clahe=True)
        assert out.shape == (224, 224, 3)

    def test_apply_params_to_mask(self):
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=np.uint8)
        cv2.circle(mask, (256, 256), 50, 255, -1)
        out, params = preprocess(img, img_size=224)
        mask_out = apply_params_to_mask(mask, params, 224)
        assert mask_out.shape == (224, 224)
        assert mask_out.max() > 0


class TestAugmentation:
    def test_augment_changes_image(self):
        rng = np.random.default_rng(42)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        aug = augment_image(img, rng)
        assert aug.shape == img.shape
        assert not np.array_equal(aug, img)


class TestUncertaintyMetrics:
    def test_temperature_scale(self):
        logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]])
        probs = temperature_scale(logits, 1.0)
        assert probs.shape == logits.shape
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_fit_temperature(self):
        logits = np.random.randn(100, 5)
        labels = np.random.randint(0, 5, 100)
        T = fit_temperature(logits, labels)
        assert T > 0

    def test_ece_score(self):
        probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.4, 0.6], [0.1, 0.9]])
        labels = np.array([0, 0, 1, 1])
        ece = ece_score(probs, labels)
        assert 0 <= ece <= 1

    def test_brier_score(self):
        probs = np.array([[0.9, 0.1], [0.1, 0.9]])
        labels = np.array([0, 1])
        bs = brier_score(probs, labels)
        assert bs >= 0

    def test_nll_score(self):
        probs = np.array([[0.9, 0.1], [0.1, 0.9]])
        labels = np.array([0, 1])
        nll = nll_score(probs, labels)
        assert nll >= 0

    def test_predictive_entropy(self):
        probs = np.array([[1.0, 0.0], [0.5, 0.5]])
        entropy = predictive_entropy(probs)
        assert entropy.shape == (2,)
        assert np.isclose(entropy[0], 0.0, atol=1e-10)
        assert np.isclose(entropy[1], 1.0, atol=1e-10)

    def test_mutual_information(self):
        mc_probs = np.random.rand(10, 4, 5)
        mc_probs = mc_probs / mc_probs.sum(axis=-1, keepdims=True)
        mi = mutual_information(mc_probs)
        assert mi.shape == (4,)
        assert np.all(mi >= -1e-6)

    def test_bootstrap_ci(self):
        vals = np.random.randn(100)
        lo, hi = bootstrap_ci(vals, n_resamples=100, rng=np.random.default_rng(42))
        assert lo <= hi


class TestLesionMetrics:
    def test_dice_perfect(self):
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[30:70, 30:70] = 1
        assert dice_coefficient(mask, mask) == 1.0

    def test_dice_zero(self):
        mask1 = np.zeros((100, 100), dtype=np.float32)
        mask1[30:50, 30:50] = 1
        mask2 = np.zeros((100, 100), dtype=np.float32)
        mask2[70:90, 70:90] = 1
        assert dice_coefficient(mask1, mask2) == 0.0

    def test_iou(self):
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[30:70, 30:70] = 1
        assert iou_score(mask, mask) == 1.0

    def test_precision_recall(self):
        pred = np.zeros((100, 100), dtype=np.float32)
        pred[30:70, 30:70] = 1
        target = np.zeros((100, 100), dtype=np.float32)
        target[40:80, 40:80] = 1
        p, r = precision_recall(pred, target)
        assert 0 <= p <= 1
        assert 0 <= r <= 1

    def test_pointing_game(self):
        heatmap = np.zeros((100, 100), dtype=np.float32)
        heatmap[50, 50] = 1.0
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[50, 50] = 1
        assert pointing_game(heatmap, mask, tolerance=0) is True
        mask[50, 50] = 0
        mask[49, 50] = 1
        assert pointing_game(heatmap, mask, tolerance=1) is True

    def test_binarize_threshold(self):
        heatmap = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        bin_map = binarize_heatmap(heatmap, "threshold", threshold=0.5)
        assert bin_map.dtype == np.float32
        assert np.sum(bin_map) == 50

    def test_binarize_top_fraction(self):
        heatmap = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
        bin_map = binarize_heatmap(heatmap, "top_fraction", fraction=0.1)
        assert np.sum(bin_map) == 10

    def test_random_baseline(self):
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[30:70, 30:70] = 1
        rng = np.random.default_rng(42)
        rand = random_baseline(mask, rng)
        assert rand.shape == mask.shape
        assert np.sum(rand) > 0

    def test_center_gaussian_baseline(self):
        mask = np.zeros((100, 100), dtype=np.float32)
        base = center_gaussian_baseline(mask)
        assert base.shape == mask.shape
        assert base[50, 50] == 1.0


class TestConfig:
    def test_load_smoke_config(self):
        cfg = load_config("configs/smoke.yaml")
        assert isinstance(cfg, Config)
        assert cfg.dr.img_size == 160
        assert cfg.dr.pretrained is False
        assert cfg.dr.epochs == 2


class TestUtils:
    def test_seed_everything(self):
        seed_everything(123)
        a = np.random.rand()
        b = torch.rand(1).item()
        seed_everything(123)
        assert np.random.rand() == a
        assert torch.rand(1).item() == b

    def test_get_device(self):
        dev = get_device("cpu")
        assert dev.type == "cpu"

    def test_run_info(self):
        info = run_info(None, {"extra": "data"})
        assert "versions" in info
        assert "git_hash" in info
        assert info["extra"] == "data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])