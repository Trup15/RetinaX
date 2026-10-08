#!/usr/bin/env python
"""
RetinaXPipeline: quality gate -> DR -> calibrated uncertainty -> referral -> optional XAI
Matches the API contract in backend_integration.md §2.
"""
import argparse
import json
import uuid
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retinax.config import load_config
from retinax.preprocessing.preprocess import preprocess, apply_params_to_mask, PreprocessParams
from retinax.uncertainty.metrics import temperature_scale, predictive_entropy
from retinax.utils import get_device, setup_logging


class RetinaXPipeline:
    """End-to-end screening pipeline."""

    def __init__(self, cfg, device, quality_model, dr_model, 
                 calib_data, quality_threshold, quality_transform=None):
        self.cfg = cfg
        self.device = device
        self.quality_model = quality_model
        self.dr_model = dr_model
        self.calib_data = calib_data
        self.quality_threshold = quality_threshold
        self.quality_transform = quality_transform
        
        self.dr_loaded = dr_model is not None
        self.quality_loaded = quality_model is not None
        self.calib_loaded = calib_data is not None
        self.quality_thresh_loaded = quality_threshold is not None

        # Class names
        self.stage_names = cfg.dr.class_names
        self.quality_class_names = cfg.quality.class_names

    @classmethod
    def load(cls, cfg):
        """Load pipeline from config and artifacts."""
        log = setup_logging("retinax.pipeline")
        device = get_device(cfg.device)
        
        artifacts_dir = Path(cfg.data.artifacts_dir)
        
        # Load quality model
        quality_model = None
        quality_ckpt_path = artifacts_dir / "quality_mnv3s.pt"
        if quality_ckpt_path.exists():
            try:
                ckpt = torch.load(quality_ckpt_path, map_location=device, weights_only=False)
                quality_model = timm.create_model(
                    cfg.quality.arch,
                    pretrained=False,
                    num_classes=cfg.quality.num_classes,
                ).to(device)
                quality_model.load_state_dict(ckpt["state_dict"])
                quality_model.eval()
                log.info("Quality model loaded")
            except Exception as e:
                log.warning(f"Failed to load quality model: {e}")
        else:
            log.warning("Quality model not found at artifacts/quality_mnv3s.pt")

        # Load DR model
        dr_model = None
        dr_ckpt_path = artifacts_dir / "dr_effb3_fold0.pt"
        if dr_ckpt_path.exists():
            try:
                ckpt = torch.load(dr_ckpt_path, map_location=device, weights_only=False)
                dr_model = timm.create_model(
                    cfg.dr.arch,
                    pretrained=False,
                    num_classes=cfg.dr.num_classes,
                    drop_rate=cfg.dr.drop_rate,
                    drop_path_rate=cfg.dr.drop_path_rate,
                ).to(device)
                dr_model.load_state_dict(ckpt["state_dict"])
                dr_model.eval()
                log.info("DR model loaded")
            except Exception as e:
                log.warning(f"Failed to load DR model: {e}")
        else:
            log.warning("DR model not found at artifacts/dr_effb3_fold0.pt")

        # Load calibration
        calib_data = None
        calib_path = artifacts_dir / "calibration.json"
        if calib_path.exists():
            with open(calib_path) as f:
                calib_data = json.load(f)
            log.info(f"Calibration loaded: T={calib_data['temperature']:.4f}")
        else:
            log.warning("Calibration not found at artifacts/calibration.json")

        # Load quality threshold
        quality_threshold = None
        thresh_path = artifacts_dir / "quality_threshold.json"
        if thresh_path.exists():
            with open(thresh_path) as f:
                thresh_data = json.load(f)
                quality_threshold = thresh_data.get("threshold", 0.5)
            log.info(f"Quality threshold loaded: {quality_threshold:.4f}")
        else:
            log.warning("Quality threshold not found at artifacts/quality_threshold.json")

        return cls(cfg, device, quality_model, dr_model, calib_data, quality_threshold)

    def screen(self, img_bgr, xai_method=None, mc_dropout=False) -> dict:
        """Run full screening pipeline on a BGR image."""
        screening_id = str(uuid.uuid4())
        
        # Preprocess for DR model (same crop, different resize)
        dr_img, params = preprocess(img_bgr, img_size=self.cfg.dr.img_size, clahe=self.cfg.dr.clahe)
        dr_tensor = self._to_tensor(dr_img, self.cfg.dr.mean, self.cfg.dr.std).unsqueeze(0).to(self.device)
        
        # Quality gate
        quality_result = self._quality_gate(img_bgr, params)
        
        if quality_result["status"] == "UNGRADABLE":
            return self._build_response(
                screening_id=screening_id,
                inference_mode="quality_only",
                quality=quality_result,
                classification=None,
                uncertainty=None,
                clinical_action="RECAPTURE_IMAGE",
                reason="UNGRADABLE",
                explainability=None,
            )
        
        if not self.dr_loaded:
            return self._build_response(
                screening_id=screening_id,
                inference_mode="quality_only",
                quality=quality_result,
                classification=None,
                uncertainty=None,
                clinical_action="MODEL_NOT_LOADED",
                reason="MODEL_NOT_LOADED",
                explainability=None,
            )
        
        # DR classification
        with torch.inference_mode():
            logits = self.dr_model(dr_tensor)
        
        # Calibrated probabilities
        if self.calib_loaded:
            T = self.calib_data["temperature"]
            probs = temperature_scale(logits.cpu().numpy(), self.calib_data["temperature"])
            probs = torch.from_numpy(probs).to(self.device)
        else:
            probs = F.softmax(logits / 1.0, dim=-1)
        
        pred_class = int(probs.argmax(dim=1).item())
        raw_probs = F.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
        cal_probs = probs.cpu().numpy()[0].tolist()
        
        # Uncertainty
        entropy_norm = float(predictive_entropy(np.array([cal_probs]))[0])
        
        # Referral decision
        tau = 0.0
        is_uncertain = False
        if self.calib_loaded and "tau" in self.calib_data:
            tau = self.calib_data.get("tau", 0.0)
        # For now, compute entropy threshold inline (would come from calibration in production)
        entropy_threshold = self.cfg.referral.target_rate  # placeholder
        
        # Simple entropy-based referral
        is_uncertain = entropy_norm > entropy_threshold
        clinical_action = "SPECIALIST_REFERRAL" if is_uncertain else "ACCEPT_GRADE"
        reason = "HIGH_UNCERTAINTY" if is_uncertain else None
        
        # XAI (optional)
        explainability = None
        if xai_method and xai_method != "NONE":
            explainability = self._generate_xai(dr_tensor, pred_class, xai_method)
        
        return self._build_response(
            screening_id=screening_id,
            inference_mode="model",
            quality=quality_result,
            classification={
                "predicted_stage": pred_class,
                "stage_name": self.stage_names[pred_class],
                "raw_probabilities": raw_probs,
                "calibrated_probabilities": cal_probs,
            },
            uncertainty={
                "method": "entropy",
                "temperature": float(self.calib_data.get("temperature", 1.0)) if self.calib_loaded else 1.0,
                "entropy_norm": entropy_norm,
                "tau": float(tau),
                "is_uncertain": is_uncertain,
            },
            clinical_action=clinical_action,
            reason=reason,
            explainability=explainability,
        )

    def _quality_gate(self, img_bgr, dr_params: PreprocessParams) -> dict:
        """Run quality gate on image."""
        # Preprocess for quality model (same crop, quality model's img_size)
        if self.quality_loaded:
            qual_img, _ = preprocess(img_bgr, img_size=self.cfg.quality.img_size, clahe=False)
            qual_tensor = self._to_tensor(qual_img, self.cfg.quality.mean, self.cfg.quality.std).unsqueeze(0).to(self.device)
            
            with torch.inference_mode():
                logits = self.quality_model(qual_tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            # Quality model: index 0=gradable, 1=ungradable
            p_ungradable = float(probs[1])
            threshold = self.quality_threshold if self.quality_thresh_loaded else 0.5
            status = "UNGRADABLE" if p_ungradable >= threshold else "GRADABLE"
            recommended_action = "RECAPTURE_IMAGE" if status == "UNGRADABLE" else "NONE"
            
            return {
                "status": status,
                "p_ungradable": p_ungradable,
                "threshold": float(threshold),
                "hints": self._compute_hints(img_bgr),
                "recommended_action": recommended_action,
            }
        else:
            # Fallback: heuristic quality check
            hints = self._compute_hints(img_bgr)
            # Simple Laplacian variance fallback
            threshold = 0.5  # placeholder
            p_ungradable = 0.0 if hints["laplacian_variance"] > 50 else 1.0
            status = "UNGRADABLE" if p_ungradable >= threshold else "GRADABLE"
            
            return {
                "status": status,
                "p_ungradable": p_ungradable,
                "threshold": float(threshold),
                "hints": hints,
                "recommended_action": "RECAPTURE_IMAGE" if status == "UNGRADABLE" else "NONE",
            }

    def _compute_hints(self, img_bgr) -> dict:
        """Compute image quality hints."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Laplacian variance (sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = float(laplacian.var())
        
        # Illumination uniformity (mean/std of central region)
        h, w = gray.shape
        central = gray[h//4:3*h//4, w//4:3*w//4]
        illumination_uniformity = float(central.std() / (central.mean() + 1e-6))
        
        # FOV coverage (fraction of non-black pixels)
        fov_coverage = float(np.mean(gray > 10))
        
        return {
            "laplacian_variance": laplacian_var,
            "illumination_uniformity": illumination_uniformity,
            "fov_coverage": fov_coverage,
        }

    def _to_tensor(self, img_bgr, mean, std) -> torch.Tensor:
        """Convert BGR image to normalized tensor."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_norm = (img_rgb.astype(np.float32) / 255.0 - mean) / std
        return torch.from_numpy(img_norm).permute(2, 0, 1).float()

    def _generate_xai(self, img_tensor, target_class, method) -> dict:
        """Generate XAI heatmap (placeholder)."""
        # Placeholder - would integrate GradCAM++/ScoreCAM/IG
        return {
            "method": method,
            "target_class": target_class,
            "heatmap_png_base64": "",
        }

    def _build_response(self, screening_id, inference_mode, quality, classification,
                       uncertainty, clinical_action, reason, explainability) -> dict:
        """Build standardized response dict."""
        model_provenance = None
        if self.dr_loaded:
            # Get from DR model meta
            model_provenance = "aptos_dev_fold0"  # Would read from checkpoint meta
        elif self.quality_loaded:
            model_provenance = "quality_only"
        else:
            model_provenance = None

        return {
            "screening_id": screening_id,
            "inference_mode": inference_mode,
            "model_provenance": model_provenance,
            "quality": quality,
            "classification": classification,
            "uncertainty": uncertainty,
            "clinical_action": clinical_action,
            "reason": reason,
            "explainability": explainability,
        }


def main():
    parser = argparse.ArgumentParser(description="Run RetinaX pipeline on image")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--image", type=str, required=True, help="Path to fundus image")
    parser.add_argument("--xai", type=str, default=None, choices=["NONE", "GRAD_CAM_PP", "SCORE_CAM", "INTEGRATED_GRADIENTS"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipe = RetinaXPipeline.load(cfg)
    
    img = cv2.imread(args.image)
    if img is None:
        print(f"Failed to read image: {args.image}")
        sys.exit(1)
    
    result = pipe.screen(img, xai_method=args.xai)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()