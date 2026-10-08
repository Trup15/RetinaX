#!/usr/bin/env python
"""
Calibrate temperature on APTOS validation fold of primary model.
Per ml_protocol.md §7.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retinax.config import load_config
from retinax.uncertainty.metrics import fit_temperature, ece_score, brier_score, nll_score, temperature_scale
from retinax.utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Calibrate temperature on validation fold")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--val-npz", type=str, default="outputs/predictions/aptos_fold0_val.npz")
    parser.add_argument("--out-json", type=str, default="artifacts/calibration.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging("retinax.uncertainty.calibrate")

    # Load validation predictions
    data = np.load(args.val_npz, allow_pickle=True)
    val_logits = data["logits"]
    val_labels = data["labels"]

    print(f"Loaded {len(val_labels)} validation samples")
    print(f"Labels: {val_labels}")

    # Fit temperature
    T = fit_temperature(val_logits, val_labels)
    print(f"Fitted temperature T = {T:.4f}")

    # Compute metrics before/after
    val_probs_before = torch.softmax(torch.from_numpy(val_logits), dim=-1).numpy()
    val_probs_after = temperature_scale(val_logits, T)

    ece_before = ece_score(val_probs_before, val_labels)
    ece_after = ece_score(val_probs_after, val_labels)
    brier_before = brier_score(val_probs_before, val_labels)
    brier_after = brier_score(val_probs_after, val_labels)
    nll_before = nll_score(val_probs_before, val_labels)
    nll_after = nll_score(val_probs_after, val_labels)

    print(f"ECE: before={ece_before:.4f}, after={ece_after:.4f}")
    print(f"Brier: before={brier_before:.4f}, after={brier_after:.4f}")
    print(f"NLL: before={nll_before:.4f}, after={nll_after:.4f}")

    # Save calibration
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    calib_data = {
        "temperature": float(T),
        "fitted_on": "aptos_fold0_val",
        "n_val": int(len(val_labels)),
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "brier_before": float(brier_before),
        "brier_after": float(brier_after),
        "nll_before": float(nll_before),
        "nll_after": float(nll_after),
        "seed": cfg.seed,
    }
    with open(out_path, "w") as f:
        json.dump(calib_data, f, indent=2)
    print(f"\nSaved calibration to {out_path}")


if __name__ == "__main__":
    main()