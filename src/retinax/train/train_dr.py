#!/usr/bin/env python
"""
Train DR classifier (EfficientNet-B3) on APTOS folds.
Per ml_protocol.md §3.
"""
import argparse
import json
import yaml
from pathlib import Path
import sys

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retinax.data.datasets import DRDataset
from retinax.config import load_config
from retinax.utils import seed_everything, get_device, write_run_info, setup_logging


def get_class_weights(labels: np.ndarray, num_classes: int = 5, method: str = "sqrt") -> torch.Tensor:
    """Compute class weights for imbalanced data."""
    counts = np.bincount(labels, minlength=num_classes)
    if method == "sqrt":
        weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    elif method == "inverse":
        weights = 1.0 / np.maximum(counts, 1)
    else:
        weights = np.ones(num_classes)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


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
    all_logits = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Val", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.array(all_labels),
        np.array(all_preds),
        np.concatenate(all_logits, axis=0) if all_logits else np.array([]),
    )


def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    """Compute QWK."""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def main():
    parser = argparse.ArgumentParser(description="Train DR classifier")
    parser.add_argument("--fold", type=int, required=True, help="Fold index (0-4)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--data-root", type=str, default="cache", help="Data root for cache (where cache/ subdir lives)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    log = setup_logging("retinax.train.train_dr")
    log.info(f"Training fold {args.fold} on {device}")

    # Load splits
    splits_df = pd.read_csv("metadata/aptos_splits.csv")
    fold_df = splits_df[splits_df["fold"] == args.fold].reset_index(drop=True)
    train_df = splits_df[(splits_df["split"] == "dev") & (splits_df["fold"] != args.fold)].reset_index(drop=True)
    val_df = fold_df

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"Train label dist: {train_df['label'].value_counts().sort_index().to_dict()}")
    print(f"Val label dist: {val_df['label'].value_counts().sort_index().to_dict()}")

    # Build datasets
    train_ds = DRDataset(
        cache_csv="metadata/aptos_cache.csv",
        data_root=args.data_root,
        img_size=cfg.dr.img_size,
        mean=cfg.dr.mean,
        std=cfg.dr.std,
        train=True,
        clahe=cfg.dr.clahe,
    )
    val_ds = DRDataset(
        cache_csv="metadata/aptos_cache.csv",
        data_root=args.data_root,
        img_size=cfg.dr.img_size,
        mean=cfg.dr.mean,
        std=cfg.dr.std,
        train=False,
        clahe=cfg.dr.clahe,
    )

    # Create subsets
    train_indices = [i for i, row in train_ds.cache_df.iterrows() if row["image_path"] in train_df["image_path"].values]
    val_indices = [i for i, row in val_ds.cache_df.iterrows() if row["image_path"] in val_df["image_path"].values]

    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(val_ds, val_indices)

    import sys

    # Use 0 workers on Windows to avoid multiprocessing issues
    NUM_WORKERS = 0 if sys.platform == "win32" else cfg.data.num_workers
    PIN_MEMORY = cfg.data.pin_memory and (sys.platform != "win32" or torch.cuda.is_available())

    train_loader = DataLoader(
        train_subset, batch_size=cfg.dr.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=cfg.dr.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # Model
    model = timm.create_model(
        cfg.dr.arch,
        pretrained=cfg.dr.pretrained,
        num_classes=cfg.dr.num_classes,
        drop_rate=cfg.dr.drop_rate,
        drop_path_rate=cfg.dr.drop_path_rate,
    ).to(device)

    # Class weights
    if cfg.dr.loss == "ce_sqrtw":
        train_labels = train_df["label"].values
        class_weights = get_class_weights(train_labels, cfg.dr.num_classes, "sqrt").to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif cfg.dr.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=cfg.dr.lr, weight_decay=cfg.dr.weight_decay)

    # Scheduler with warmup
    def lr_lambda(epoch):
        if epoch < cfg.dr.warmup_epochs:
            return float(epoch + 1) / cfg.dr.warmup_epochs
        progress = (epoch - cfg.dr.warmup_epochs) / (cfg.dr.epochs - cfg.dr.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume
    start_epoch = 0
    best_qwk = -1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_qwk = ckpt.get("best_qwk", -1)
        log.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    output_dir = Path(cfg.data.outputs_dir) / f"train_dr_fold{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(cfg.data.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, cfg.dr.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, cfg.dr.grad_clip)
        val_loss, val_acc, val_labels, val_preds, val_logits = validate(model, val_loader, criterion, device)

        val_qwk = quadratic_weighted_kappa(val_labels, val_preds)
        val_balanced_acc = np.mean([np.mean(val_preds[val_labels == c] == c) for c in range(cfg.dr.num_classes)])

        log.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                 f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_qwk={val_qwk:.4f}, "
                 f"val_bal_acc={val_balanced_acc:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()

        # Save best
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            ckpt_path = artifacts_dir / f"dr_effb3_fold{args.fold}.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_qwk": best_qwk,
                "meta": {
                    "arch": cfg.dr.arch,
                    "num_classes": cfg.dr.num_classes,
                    "img_size": cfg.dr.img_size,
                    "mean": cfg.dr.mean,
                    "std": cfg.dr.std,
                    "clahe": cfg.dr.clahe,
                    "class_names": cfg.dr.class_names,
                    "trained_on": "aptos_dev_fold{}".format(args.fold) if cfg.dr.pretrained else "synthetic_smoke",
                    "val_qwk": best_qwk,
                    "seed": cfg.seed,
                    "timm_version": timm.__version__,
                }
            }, ckpt_path)
            log.info(f"Saved best model to {ckpt_path} (QWK={best_qwk:.4f})")

        # Save predictions for calibration
        if epoch == cfg.dr.epochs - 1:
            pred_path = Path(cfg.data.outputs_dir) / "predictions" / f"aptos_fold{args.fold}_val.npz"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(pred_path,
                     ids=np.array(val_df["image_path"].values),
                     labels=val_labels,
                     logits=val_logits)

    # Save run info
    write_run_info(output_dir, {
        "fold": args.fold,
        "epochs": cfg.dr.epochs,
        "best_qwk": best_qwk,
        "config": cfg.__dict__,
    })

    print(f"Training complete. Best QWK: {best_qwk:.4f}")


if __name__ == "__main__":
    main()