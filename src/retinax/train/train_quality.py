#!/usr/bin/env python
"""
Train quality model (MobileNetV3-Small) on DDR + synthetic ungradable.
Per ml_protocol.md §5.
"""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retinax.data.datasets import QualityDataset
from retinax.config import load_config
from retinax.utils import seed_everything, get_device, write_run_info, setup_logging


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels, _ in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Val", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]  # Prob of ungradable (class 1)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def find_threshold_for_sensitivity(labels, probs, target_sensitivity=0.95):
    """Find threshold on valid set to achieve target sensitivity for ungradable class."""
    # labels: 0=gradable, 1=ungradable
    # We want P(ungradable | label=1) >= target_sensitivity
    # i.e., recall for class 1
    ungradable_mask = labels == 1
    if not np.any(ungradable_mask):
        return 0.5

    ungradable_probs = probs[ungradable_mask]
    # Sort thresholds
    thresholds = np.linspace(0, 1, 1001)
    best_thresh = 0.5
    for thresh in thresholds:
        pred = (probs >= thresh).astype(int)
        sensitivity = np.sum((pred == 1) & ungradable_mask) / np.sum(ungradable_mask)
        if sensitivity >= target_sensitivity:
            best_thresh = thresh
            break
    return best_thresh


def main():
    parser = argparse.ArgumentParser(description="Train quality model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-root", type=str, default="cache", help="Data root for cache (where cache/ subdir lives)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    log = setup_logging("retinax.train.train_quality")
    log.info(f"Training quality model on {device}")

    # Load quality metadata (balanced gradable/ungradable)
    meta_df = pd.read_csv("metadata/quality_meta.csv")
    print(f"Quality meta: {len(meta_df)} total")
    print(f"Gradable dist: {meta_df['gradable'].value_counts().to_dict()}")

    # Split: use DDR test as validation, train+valid as train
    # For smoke test, use 80/20 split
    train_df = meta_df.sample(frac=0.8, random_state=cfg.seed).reset_index(drop=True)
    val_df = meta_df.drop(train_df.index).reset_index(drop=True)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"Train gradable: {train_df['gradable'].value_counts().to_dict()}")
    print(f"Val gradable: {val_df['gradable'].value_counts().to_dict()}")

    # Build datasets
    train_ds = QualityDataset(
        cache_csv="metadata/quality_cache.csv",
        data_root=args.data_root,
        img_size=cfg.quality.img_size,
        mean=cfg.quality.mean,
        std=cfg.quality.std,
        train=True,
    )
    val_ds = QualityDataset(
        cache_csv="metadata/quality_cache.csv",
        data_root=args.data_root,
        img_size=cfg.quality.img_size,
        mean=cfg.quality.mean,
        std=cfg.quality.std,
        train=False,
    )

    # Create subsets
    train_indices = [i for i, row in train_ds.cache_df.iterrows() if row["image_path"] in train_df["image_path"].values]
    val_indices = [i for i, row in val_ds.cache_df.iterrows() if row["image_path"] in val_df["image_path"].values]

    train_subset = torch.utils.data.Subset(train_ds, train_indices)
    val_subset = torch.utils.data.Subset(val_ds, val_indices)

    import sys

    # Use 0 workers on Windows to avoid multiprocessing issues
    NUM_WORKERS = 0 if sys.platform == "win32" else cfg.data.num_workers
    PIN_MEMORY = cfg.data.pin_memory and (sys.platform != "win32" or torch.cuda.is_available())

    train_loader = DataLoader(
        train_subset, batch_size=cfg.quality.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=cfg.quality.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # Model
    model = timm.create_model(
        cfg.quality.arch,
        pretrained=cfg.quality.pretrained,
        num_classes=cfg.quality.num_classes,
    ).to(device)

    # Weighted CE for imbalance (if any)
    train_labels = train_df["gradable"].map({1: 0, 0: 1}).values  # 0=gradable, 1=ungradable
    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float32).to(device)
    class_weights = class_weights / class_weights.sum() * 2
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=cfg.quality.lr, weight_decay=cfg.quality.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.quality.epochs, eta_min=1e-6)

    # Training loop
    output_dir = Path(cfg.data.outputs_dir) / "train_quality"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(cfg.data.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    best_auc = 0
    for epoch in range(cfg.quality.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_labels, val_preds, val_probs = validate(model, val_loader, criterion, device)

        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5
        cm = confusion_matrix(val_labels, val_preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for ungradable (class 1)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        log.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                 f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_auc={val_auc:.4f}, "
                 f"sens={sensitivity:.4f}, spec={specificity:.4f}")

        scheduler.step()

        # Save best by AUC
        if val_auc > best_auc:
            best_auc = val_auc
            ckpt_path = artifacts_dir / "quality_mnv3s.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_auc": best_auc,
                "meta": {
                    "arch": cfg.quality.arch,
                    "num_classes": cfg.quality.num_classes,
                    "img_size": cfg.quality.img_size,
                    "mean": cfg.quality.mean,
                    "std": cfg.quality.std,
                    "class_names": cfg.quality.class_names,
                    "trained_on": "ddr_synthetic_ungradable",
                    "val_auc": best_auc,
                    "seed": cfg.seed,
                    "timm_version": timm.__version__,
                }
            }, ckpt_path)
            log.info(f"Saved best model to {ckpt_path} (AUC={best_auc:.4f})")

    # Find threshold on validation for target sensitivity
    threshold = find_threshold_for_sensitivity(val_labels, val_probs, cfg.quality.target_sensitivity)
    print(f"Threshold for sensitivity >= {cfg.quality.target_sensitivity}: {threshold:.4f}")

    # Evaluate on test set (DDR test split)
    ddr_meta = pd.read_csv("metadata/ddr_meta.csv")
    test_df = ddr_meta[ddr_meta["split"] == "test"].reset_index(drop=True)
    print(f"DDR test: {len(test_df)} images (all gradable)")

    test_ds = QualityDataset(
        cache_csv="metadata/quality_cache.csv",
        data_root=args.data_root,
        img_size=cfg.quality.img_size,
        mean=cfg.quality.mean,
        std=cfg.quality.std,
        train=False,
    )
    test_indices = [i for i, row in test_ds.cache_df.iterrows() if row["image_path"] in test_df["image_path"].values]

    if len(test_indices) > 0:
        test_subset = torch.utils.data.Subset(test_ds, test_indices)
        test_loader = DataLoader(test_subset, batch_size=cfg.quality.batch_size, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        test_loss, test_acc, test_labels, test_preds, test_probs = validate(model, test_loader, criterion, device)
        # On test set (all gradable), we expect low false positive rate
        fp_rate = np.mean(test_preds == 1)  # All should be 0 (gradable)
        print(f"Test FPR (should be low): {fp_rate:.4f}")
    else:
        print("Test set images not in quality cache - skipping test evaluation")
        fp_rate = None

    # Save threshold
    threshold_path = artifacts_dir / "quality_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({
            "threshold": float(threshold),
            "target_sensitivity": cfg.quality.target_sensitivity,
            "val_sensitivity": float(sensitivity),
            "val_specificity": float(specificity),
            "val_auc": float(best_auc),
            "test_fpr": float(fp_rate) if fp_rate is not None else None,
            "fitted_on": "quality_validation_split",
        }, f, indent=2)
    print(f"Saved threshold to {threshold_path}")

    # Save run info
    write_run_info(output_dir, {
        "epochs": cfg.quality.epochs,
        "best_auc": best_auc,
        "threshold": threshold,
        "test_fpr": fp_rate,
        "config": cfg.__dict__,
    })

    print(f"Quality training complete. Best AUC: {best_auc:.4f}, Threshold: {threshold:.4f}")


if __name__ == "__main__":
    main()