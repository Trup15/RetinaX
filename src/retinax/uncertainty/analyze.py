#!/usr/bin/env python
"""
Uncertainty analysis and referral threshold selection.
Per ml_protocol.md §8.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.stats import entropy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retinax.config import load_config
from retinax.uncertainty.metrics import (
    temperature_scale, predictive_entropy, mutual_information, 
    ece_score, brier_score, nll_score
)
from retinax.utils import setup_logging


def compute_uncertainties(logits, T=1.0, mc_probs=None):
    """Compute various uncertainty measures from logits."""
    probs = temperature_scale(logits, T)
    
    # MSP uncertainty
    msp_uncertainty = 1 - np.max(probs, axis=1)
    
    # Predictive entropy (normalized)
    pred_entropy = predictive_entropy(probs)
    
    # MC-Dropout mutual information (if available)
    mi = None
    if mc_probs is not None:
        mi = mutual_information(mc_probs)
    
    return {
        "msp": msp_uncertainty,
        "entropy": pred_entropy,
        "mi": mi,
        "probs": probs,
    }


def select_referral_threshold(entropy_vals, labels, target_rate=0.15):
    """Select threshold tau on validation for target referral rate."""
    # Sort by entropy descending (higher entropy = more uncertain)
    sorted_idx = np.argsort(entropy_vals)[::-1]
    n_refer = int(target_rate * len(entropy_vals))
    if n_refer == 0:
        n_refer = 1
    threshold = entropy_vals[sorted_idx[n_refer - 1]]
    return threshold


def evaluate_referral(entropy_vals, labels, probs, threshold, target_rate=0.15):
    """Evaluate referral at given threshold."""
    refer_mask = entropy_vals >= threshold
    referral_rate = np.mean(refer_mask)
    
    # Selective accuracy (accuracy on retained)
    retain_mask = ~refer_mask
    if np.sum(retain_mask) > 0:
        retain_acc = np.mean(np.argmax(probs[retain_mask], axis=1) == labels[retain_mask])
    else:
        retain_acc = 0.0
    
    # Error capture rate
    errors = (np.argmax(probs, axis=1) != labels)
    error_capture = np.mean(errors[refer_mask]) if np.sum(refer_mask) > 0 else 0.0
    
    return {
        "referral_rate": referral_rate,
        "selective_accuracy": retain_acc,
        "error_capture_rate": error_capture,
        "n_referred": int(np.sum(refer_mask)),
        "n_retained": int(np.sum(retain_mask)),
    }


def main():
    parser = argparse.ArgumentParser(description="Uncertainty analysis and referral")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--val-npz", type=str, default="outputs/predictions/aptos_fold0_val.npz")
    parser.add_argument("--calib-json", type=str, default="artifacts/calibration.json")
    parser.add_argument("--out-dir", type=str, default="outputs/uncertainty")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("retinax.uncertainty.analyze")

    # Load validation predictions
    data = np.load(args.val_npz, allow_pickle=True)
    val_logits = data["logits"]
    val_labels = data["labels"]
    val_ids = data["ids"]

    # Load calibration
    with open(args.calib_json) as f:
        calib = json.load(f)
    T = calib["temperature"]

    print(f"Loaded {len(val_labels)} validation samples")
    print(f"Temperature T = {T}")

    # Compute uncertainties
    uncertainties = compute_uncertainties(val_logits, T)
    msp_u = uncertainties["msp"]
    entropy_u = uncertainties["entropy"]
    probs = uncertainties["probs"]

    # Select referral threshold on validation
    target_rate = cfg.referral.target_rate
    tau = select_referral_threshold(entropy_u, val_labels, target_rate)
    print(f"Referral threshold tau = {tau:.4f} (target rate: {target_rate})")

    # Evaluate on validation
    results = evaluate_referral(entropy_u, val_labels, probs, tau, target_rate)
    print(f"Validation: referral_rate={results['referral_rate']:.3f}, "
          f"selective_acc={results['selective_accuracy']:.3f}, "
          f"error_capture={results['error_capture_rate']:.3f}")

    # Save outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-sample uncertainties
    np.savez(out_dir / "val_uncertainty.npz",
             ids=val_ids,
             labels=val_labels,
             logits=val_logits,
             probs=probs,
             msp=msp_u,
             entropy=entropy_u,
             threshold=tau)

    # Save summary
    summary = {
        "threshold": float(tau),
        "target_rate": target_rate,
        "method": "entropy",
        "validation": results,
    }
    with open(out_dir / "uncertainty_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved uncertainty outputs to {out_dir}")


if __name__ == "__main__":
    main()