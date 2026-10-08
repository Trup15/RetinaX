#!/usr/bin/env python
"""
Prepare DDR dataset canonical metadata CSVs.
Handles both:
- Current structure: train/ and test/ subfolders with flat label CSV
- Original DDR structure: flat image folder + DR_grading.xlsx + official split files (train.txt, valid.txt, test.txt)
"""
import argparse
import pandas as pd
from pathlib import Path
import sys
import shutil


def find_label_file(data_root: Path) -> Path:
    """Find DDR label file in common locations."""
    candidates = [
        data_root / "DR_grading.xlsx",
        data_root / "DR_grading.csv",
        data_root / "label.xlsx",
        data_root / "label.csv",
        data_root / "ddr_labels.xlsx",
        data_root / "ddr_labels.csv",
        data_root.parent / "DR_grading.xlsx",
        data_root.parent / "DR_grading.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No DDR label file found. Tried: {[str(c) for c in candidates]}\n"
        "Download DDR dataset and ensure label file is present."
    )


def read_ddr_labels(label_path: Path) -> pd.DataFrame:
    """Read DDR labels from Excel or CSV."""
    if label_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(label_path)
    elif label_path.suffix.lower() == ".csv":
        df = pd.read_csv(label_path)
    else:
        raise ValueError(f"Unsupported label file format: {label_path.suffix}")

    cols_lower = {c.lower(): c for c in df.columns}
    img_col = None
    label_col = None

    for c in ["image", "img_name", "filename", "file_name", "id_code", "image_id", "id", "name"]:
        if c in cols_lower:
            img_col = cols_lower[c]
            break
    for c in ["level", "grade", "label", "dr_level", "dr_grade", "diagnosis", "class", "target"]:
        if c in cols_lower:
            label_col = cols_lower[c]
            break

    if img_col is None or label_col is None:
        raise ValueError(
            f"Could not find image/label columns in {label_path}. "
            f"Columns: {list(df.columns)}. Expected image column (image/img_name/filename/name) "
            f"and label column (level/grade/label/diagnosis)."
        )

    df = df.rename(columns={img_col: "image_id", label_col: "label"})
    df["image_id"] = df["image_id"].astype(str)
    df["image_id"] = df["image_id"].apply(lambda x: x if "." in x else f"{x}.jpg")
    df["label"] = df["label"].astype(int)
    return df[["image_id", "label"]]


def get_official_splits(data_root: Path) -> dict:
    """Load official DDR train/valid/test splits if available."""
    split_files = {
        "train": ["train.txt", "train_list.txt", "train.csv", "train_split.txt"],
        "valid": ["valid.txt", "val.txt", "valid_list.txt", "val_list.txt", "valid.csv", "val.csv", "valid_split.txt", "val_split.txt"],
        "test": ["test.txt", "test_list.txt", "test.csv", "test_split.txt"],
    }
    splits = {}
    for split_name, candidates in split_files.items():
        for c in candidates:
            p = data_root / c
            if p.exists():
                if p.suffix == ".csv":
                    df = pd.read_csv(p, header=None)
                    splits[split_name] = df[0].astype(str).tolist()
                else:
                    with open(p) as f:
                        splits[split_name] = [line.strip() for line in f if line.strip()]
                print(f"  Found {split_name} split: {len(splits[split_name])} images")
                break
    return splits


def detect_dataset_structure(data_root: Path) -> str:
    """Detect whether dataset uses original (flat) or current (train/test folders) structure."""
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    
    # Check for original structure: flat images + split files
    has_split_files = any((data_root / f).exists() for f in ["train.txt", "valid.txt", "test.txt", "train_split.txt", "val_split.txt", "test_split.txt"])
    has_flat_images = any(data_root.glob("*.jpg")) or any(data_root.glob("*.png")) or any(data_root.glob("*.jpeg"))
    
    if has_split_files and has_flat_images:
        return "original"
    elif train_dir.exists() and test_dir.exists():
        return "current"
    else:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Prepare DDR metadata CSVs")
    parser.add_argument("--data-root", type=str, default="data/ddr_original",
                        help="Root folder containing DDR dataset (original or current structure)")
    parser.add_argument("--label-file", type=str, default=None,
                        help="Explicit path to label file (Excel/CSV)")
    parser.add_argument("--out-dir", type=str, default="metadata",
                        help="Output directory for metadata CSVs")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detect structure
    structure = detect_dataset_structure(data_root)
    print(f"Detected dataset structure: {structure}")

    if structure == "original":
        # Original DDR: flat images folder + split files
        image_dir = data_root
        train_dir = data_root  # Not used, but kept for compatibility
        test_dir = data_root   # Not used
    elif structure == "current":
        # Current structure: train/ and test/ subfolders
        train_dir = data_root / "train"
        test_dir = data_root / "test"
        
        if not train_dir.exists() or not test_dir.exists():
            print(f"ERROR: Expected train/ and test/ directories under {data_root}")
            sys.exit(1)
        
        # Collect images from both train and test folders
        image_dirs = [train_dir, test_dir]
    else:
        print(f"ERROR: Could not detect dataset structure in {data_root}")
        print("Expected either:")
        print("  - Original: flat images + train.txt/valid.txt/test.txt")
        print("  - Current: train/ and test/ subfolders")
        sys.exit(1)

    # Find/load label file
    if args.label_file:
        label_path = Path(args.label_file)
    else:
        try:
            label_path = find_label_file(data_root)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("Use --label-file to specify path explicitly.")
            sys.exit(1)

    print(f"Reading labels from: {label_path}")
    labels_df = read_ddr_labels(label_path)
    print(f"Loaded {len(labels_df)} labeled images")
    print(f"Label distribution: {labels_df['label'].value_counts().sort_index().to_dict()}")

    # Get official splits if available
    official_splits = get_official_splits(data_root)
    if official_splits:
        print(f"Found official splits: { {k: len(v) for k, v in official_splits.items()} }")

    # Collect all image files
    all_images = []
    seen = set()
    if structure == "original":
        search_dirs = [image_dir]
    else:
        search_dirs = image_dirs
    
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp", ".BMP"]:
        for search_dir in search_dirs:
            for p in search_dir.glob(f"*{ext}"):
                key = p.name.lower()
                if key not in seen:
                    seen.add(key)
                    all_images.append(p)

    print(f"Found {len(all_images)} image files")

    # Build metadata rows
    rows = []
    matched = 0
    for img_path in all_images:
        img_name = img_path.name
        # Match label (try exact, then without extension)
        label_row = labels_df[labels_df["image_id"] == img_name]
        if label_row.empty:
            # Try without extension
            stem = img_path.stem
            label_row = labels_df[labels_df["image_id"].str.startswith(stem)]

        if label_row.empty:
            print(f"WARNING: No label found for {img_name}")
            continue

        label = int(label_row.iloc[0]["label"])
        gradable = 0 if label == 5 else 1

        # Determine split
        if official_splits:
            if img_name in official_splits.get("train", []):
                split = "train"
            elif img_name in official_splits.get("valid", []):
                split = "valid"
            elif img_name in official_splits.get("test", []):
                split = "test"
            else:
                # Fallback: if no official split matches, use folder inference
                try:
                    rel = img_path.relative_to(train_dir)
                    split = "train"
                except ValueError:
                    try:
                        rel = img_path.relative_to(test_dir)
                        split = "test"
                    except ValueError:
                        split = "unknown"
        else:
            # No official splits - infer from folder structure
            try:
                rel = img_path.relative_to(train_dir)
                split = "train"
            except ValueError:
                try:
                    rel = img_path.relative_to(test_dir)
                    split = "test"
                except ValueError:
                    split = "unknown"

        rel_path = img_path.relative_to(data_root).as_posix()

        rows.append({
            "image_path": rel_path,
            "label": label,
            "gradable": gradable,
            "split": split,
        })
        matched += 1

    print(f"Matched {matched}/{len(all_images)} images with labels")

    if matched == 0:
        print("ERROR: No images matched labels. Check label file format.")
        sys.exit(1)

    meta_df = pd.DataFrame(rows)
    meta_df = meta_df.sort_values("image_path").reset_index(drop=True)

    out_path = out_dir / "ddr_meta.csv"
    meta_df.to_csv(out_path, index=False)
    print(f"\nWritten {out_path} ({len(meta_df)} rows)")
    print(f"Label distribution:\n{meta_df['label'].value_counts().sort_index()}")
    print(f"Split distribution:\n{meta_df['split'].value_counts()}")
    print(f"Gradable distribution:\n{meta_df['gradable'].value_counts()}")


if __name__ == "__main__":
    main()