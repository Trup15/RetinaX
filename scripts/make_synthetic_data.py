import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.random import Generator, PCG64
from typing import Dict


def make_fundus_image(rng: Generator, size: tuple = (512, 512), stage: int = 0) -> np.ndarray:
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
    if stage > 1:
        n_he = rng.integers(stage * 2, stage * 5)
        for _ in range(n_he):
            hx, hy = rng.integers(cx - radius, cx + radius), rng.integers(cy - radius, cy + radius)
            if (hx - cx) ** 2 + (hy - cy) ** 2 <= radius ** 2:
                cv2.circle(img, (hx, hy), rng.integers(3, 8), (0, 100, 255), -1)
    if stage > 2:
        n_ex = rng.integers(stage, stage * 3)
        for _ in range(n_ex):
            ex, ey = rng.integers(cx - radius, cx + radius), rng.integers(cy - radius, cy + radius)
            if (ex - cx) ** 2 + (ey - cy) ** 2 <= radius ** 2:
                cv2.ellipse(img, (ex, ey), (rng.integers(5, 15), rng.integers(3, 8)), rng.integers(0, 180), 0, 360, (255, 255, 100), -1)
    if stage > 3:
        n_se = rng.integers(1, 4)
        for _ in range(n_se):
            sx, sy = rng.integers(cx - radius // 2, cx + radius // 2), rng.integers(cy - radius // 2, cy + radius // 2)
            if (sx - cx) ** 2 + (sy - cy) ** 2 <= (radius // 2) ** 2:
                cv2.circle(img, (sx, sy), rng.integers(10, 20), (255, 200, 200), -1)
    border = rng.integers(0, 30)
    if border > 0:
        img[:border, :] = 0
        img[-border:, :] = 0
        img[:, :border] = 0
        img[:, -border:] = 0
    return img


def make_lesion_masks(rng: Generator, size: tuple = (512, 512), stage: int = 0) -> Dict[str, np.ndarray]:
    h, w = size
    masks = {
        "ma": np.zeros((h, w), dtype=np.uint8),
        "he": np.zeros((h, w), dtype=np.uint8),
        "ex": np.zeros((h, w), dtype=np.uint8),
        "se": np.zeros((h, w), dtype=np.uint8),
    }
    cx, cy = w // 2, h // 2
    radius = min(cx, cy) - 20
    if stage > 0:
        n_ma = rng.integers(stage * 3, stage * 8)
        for _ in range(n_ma):
            mx, my = rng.integers(cx - radius, cx + radius), rng.integers(cy - radius, cy + radius)
            if (mx - cx) ** 2 + (my - cy) ** 2 <= radius ** 2:
                cv2.circle(masks["ma"], (mx, my), rng.integers(1, 3), 255, -1)
    if stage > 1:
        n_he = rng.integers(stage * 2, stage * 5)
        for _ in range(n_he):
            hx, hy = rng.integers(cx - radius, cx + radius), rng.integers(cy - radius, cy + radius)
            if (hx - cx) ** 2 + (hy - cy) ** 2 <= radius ** 2:
                cv2.circle(masks["he"], (hx, hy), rng.integers(3, 8), 255, -1)
    if stage > 2:
        n_ex = rng.integers(stage, stage * 3)
        for _ in range(n_ex):
            ex, ey = rng.integers(cx - radius, cx + radius), rng.integers(cy - radius, cy + radius)
            if (ex - cx) ** 2 + (ey - cy) ** 2 <= radius ** 2:
                cv2.ellipse(masks["ex"], (ex, ey), (rng.integers(5, 15), rng.integers(3, 8)), rng.integers(0, 180), 0, 360, 255, -1)
    if stage > 3:
        n_se = rng.integers(1, 4)
        for _ in range(n_se):
            sx, sy = rng.integers(cx - radius // 2, cx + radius // 2), rng.integers(cy - radius // 2, cy + radius // 2)
            if (sx - cx) ** 2 + (sy - cy) ** 2 <= (radius // 2) ** 2:
                cv2.circle(masks["se"], (sx, sy), rng.integers(10, 20), 255, -1)
    return masks


def make_ungradable_image(rng: Generator, size: tuple = (512, 512), type_: str = "blur") -> np.ndarray:
    img = make_fundus_image(rng, size, stage=0)
    if type_ == "blur":
        k = rng.integers(15, 31) | 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    elif type_ == "dark":
        img = (img * rng.uniform(0.1, 0.3)).astype(np.uint8)
    elif type_ == "bright":
        img = np.clip(img * rng.uniform(2.5, 4.0), 0, 255).astype(np.uint8)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="smoke_data")
    parser.add_argument("--n-aptos", type=int, default=80)
    parser.add_argument("--n-idrid", type=int, default=16)
    parser.add_argument("--n-ddr", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = Generator(PCG64(args.seed))
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "metadata").mkdir(parents=True, exist_ok=True)

    aptos_dir = out_root / "aptos" / "images"
    aptos_dir.mkdir(parents=True, exist_ok=True)
    aptos_rows = []
    aptos_stages = [0, 1, 2, 3, 4]
    aptos_probs = [0.4, 0.25, 0.2, 0.1, 0.05]
    for i in range(args.n_aptos):
        stage = rng.choice(aptos_stages, p=aptos_probs)
        img = make_fundus_image(rng, (512, 512), stage)
        fname = f"aptos_{i:04d}.png"
        cv2.imwrite(str(aptos_dir / fname), img)
        aptos_rows.append({"image_path": f"aptos/images/{fname}", "label": int(stage), "patient_id": f"aptos_{i:04d}"})
    pd.DataFrame(aptos_rows).to_csv(out_root / "metadata" / "aptos_meta.csv", index=False)

    idrid_dir = out_root / "idrid" / "images"
    idrid_mask_dir = out_root / "idrid" / "masks"
    idrid_dir.mkdir(parents=True, exist_ok=True)
    idrid_mask_dir.mkdir(parents=True, exist_ok=True)
    idrid_rows = []
    idrid_stages = [0, 1, 2, 3, 4]
    idrid_probs = [0.3, 0.25, 0.2, 0.15, 0.1]
    for i in range(args.n_idrid):
        stage = rng.choice(idrid_stages, p=idrid_probs)
        img = make_fundus_image(rng, (4288, 2848) if rng.random() < 0.3 else (512, 512), stage)
        fname = f"IDRiD_{i:03d}.png"
        cv2.imwrite(str(idrid_dir / fname), img)
        has_masks = 1 if i < args.n_idrid // 2 else 0
        mask_paths = {"ma_mask": "", "he_mask": "", "ex_mask": "", "se_mask": ""}
        if has_masks:
            masks = make_lesion_masks(rng, img.shape[:2], stage)
            for k, m in masks.items():
                mfname = f"{fname[:-4]}_{k}.png"
                cv2.imwrite(str(idrid_mask_dir / mfname), m)
                mask_paths[f"{k}_mask"] = f"idrid/masks/{mfname}"
        idrid_rows.append({
            "image_path": f"idrid/images/{fname}",
            "label": int(stage),
            **mask_paths,
            "has_masks": has_masks,
        })
    pd.DataFrame(idrid_rows).to_csv(out_root / "metadata" / "idrid_meta.csv", index=False)

    ddr_dir = out_root / "ddr" / "images"
    ddr_dir.mkdir(parents=True, exist_ok=True)
    ddr_rows = []
    ddr_stages = [0, 1, 2, 3, 4]
    ddr_probs = [0.35, 0.25, 0.2, 0.15, 0.05]
    for i in range(args.n_ddr):
        if i < args.n_ddr // 6:
            label = 5
            gradable = 0
            img = make_ungradable_image(rng, (512, 512), rng.choice(["blur", "dark", "bright"]))
        else:
            label = rng.choice(ddr_stages, p=ddr_probs)
            gradable = 1
            img = make_fundus_image(rng, (512, 512), label)
        fname = f"DDR_{i:04d}.png"
        cv2.imwrite(str(ddr_dir / fname), img)
        if i < args.n_ddr * 0.7:
            split = "train"
        elif i < args.n_ddr * 0.85:
            split = "valid"
        else:
            split = "test"
        ddr_rows.append({
            "image_path": f"ddr/images/{fname}",
            "label": int(label),
            "gradable": gradable,
            "split": split,
        })
    pd.DataFrame(ddr_rows).to_csv(out_root / "metadata" / "ddr_meta.csv", index=False)

    print(f"Synthetic data created at {out_root}")
    print(f"APTOS: {len(aptos_rows)} images")
    print(f"IDRiD: {len(idrid_rows)} images")
    print(f"DDR: {len(ddr_rows)} images")


if __name__ == "__main__":
    main()