#!/usr/bin/env python
"""
Generate synthetic ungradable images for quality model training.
Creates blur/dark/bright variants and appends to DDR metadata.
"""
import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.random import Generator, PCG64
from typing import Dict, List


def make_fundus_image(rng: Generator, size: tuple = (512, 512), stage: int = 0) -> np.ndarray:
    """Generate a synthetic fundus image (same as make_synthetic_data.py)."""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    radius = min(cx, cy) - 20
    y, x = np.ogrid[:h, :w]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    bg_color = rng.integers(30, 60, size=3)
    img[mask] = bg_color
    for _ in range(rng.integers(5, 15)):
        vx, vy = rng.integers(0, w), rng.integers(0, h)
        vr = rng.integers(2, 8)
        color = rng.integers(80, 150, size=3)
        cv2.circle(img, (vx, vy), vr, color.tolist(), -1)
    if stage > 0:
        n_ma = rng.integers(stage * 3, stage * 8)
        for _ in range(n_ma):
            mx, my = rng.integers(cx - radius, cx + radius), rng.integers(cy - radius, cy + radius)
            if (mx - cx) ** 2 + (my - cy) ** 2 <= radius ** 2:
                cv2.circle(img, (mx, my), rng.integers(1, 3), (0, 0, 200), -1)
    border = rng.integers(0, 30)
    if border > 0:
        img[:border, :] = 0; img[-border:, :] = 0; img[:, :border] = 0; img[:, -border:] = 0
    return img


def make_ungradable_image(rng: Generator, size: tuple = (512, 512), type_: str = "blur") -> np.ndarray:
    """Create ungradable image from a normal fundus."""
    img = make_fundus_image(rng, size, stage=0)
    if type_ == "blur":
        k = rng.integers(15, 31) | 1  # Odd kernel 15-31
        img = cv2.GaussianBlur(img, (k, k), 0)
    elif type_ == "dark":
        img = (img * rng.uniform(0.1, 0.3)).astype(np.uint8)
    elif type_ == "bright":
        img = np.clip(img * rng.uniform(2.5, 4.0), 0, 255).astype(np.uint8)
    return img


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ungradable images for quality model")
    parser.add_argument("--n", type=int, default=1000, help="Number of ungradable images to generate")
    parser.add_argument("--out-dir", type=str, default="smoke_data/DR_grading/synthetic_ungradable", help="Output directory")
    parser.add_argument("--meta-out", type=str, default="metadata/synthetic_ungradable.csv", help="Output metadata CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"], help="Split to assign")
    parser.add_argument("--size", type=int, default=512, help="Image size")
    args = parser.parse_args()

    rng = Generator(PCG64(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    types = ["blur", "dark", "bright"]
    type_weights = [0.5, 0.25, 0.25]  # Blur most common in practice

    rows = []
    for i in range(args.n):
        t = rng.choice(types, p=type_weights)
        img = make_ungradable_image(rng, (args.size, args.size), t)
        fname = f"synth_ungradable_{i:05d}_{t}.jpg"
        cv2.imwrite(str(out_dir / fname), img)
        rows.append({
            "image_path": f"synthetic_ungradable/{fname}",
            "label": 5,  # Class 5 = ungradable
            "gradable": 0,
            "split": args.split,
        })

    meta_df = pd.DataFrame(rows)
    Path(args.meta_out).parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(args.meta_out, index=False)
    print(f"Generated {args.n} synthetic ungradable images in {out_dir}")
    print(f"Metadata saved to {args.meta_out}")
    print(f"Type distribution: {meta_df.groupby(meta_df['image_path'].str.split('_').str[-1].str.replace('.jpg', '')).size().to_dict()}")


if __name__ == "__main__":
    main()