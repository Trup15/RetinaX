# Data Specification — RetinaX (improved)

## 1. Zero-download smoke data (use this first)
`reference/make_synthetic_data.py` (copy to `scripts/`) builds small stand-ins for APTOS and IDRiD **and** the canonical
metadata CSVs the pipeline consumes:
```bash
python scripts/make_synthetic_data.py --out smoke_data --n-aptos 80 --n-idrid 16
```
It has black borders, varying image sizes, stage-dependent lesions, per-lesion masks for IDRiD-like images. It is **only** for proving the code runs. Results on it mean nothing and must never be reported.

## 2. Official datasets (human downloads; see README_FIRST "Actions only the human can do")
| Dataset | Role | Notes to verify after download |
|---|---|---|
| APTOS 2019 (Kaggle `aptos2019-blindness-detection`) | train / val / internal test / calibration | Only `train.csv` (3,662 labelled) is usable; `test.csv` is unlabelled. `id_code`, `diagnosis` 0–4. Images vary in size; many have black borders. |
| IDRiD (IEEE Dataport DOI 10.21227/H25W98, or a Kaggle mirror) | external grading (516) + lesion validation (segmentation subset) | Grading CSV has image name, retinopathy grade, macular-edema risk. Masks (MA, HE, EX, SE; optionally OD) exist only for the segmentation subset (reported 81 images). Match by file stem. |
| DDR (Kaggle `mariaherrerot/ddrdataset`) | quality model (train/valid lists) + external DR evaluation (test list) | **12,524 images total: 9,378 train + 3,146 test**. Grades 0–4 (no class 5 ungradable in this split). Label file: `DR_grading.csv` with `id_code`, `diagnosis`. Folder structure: `DR_grading/train/`, `DR_grading/test/`. |
| EyePACS | optional, later | Not needed for the first build. |
Kaggle mirrors differ from the official layouts. The agent lists the folder first, then writes `prepare_*.py`.
If a download requires credentials the agent does not have, it **stops and asks**; it never substitutes synthetic data for real results.

## 3. Canonical metadata CSVs (the only thing downstream code reads)
Paths are relative to the data root (`RETINAX_DATA_ROOT`, default `./data`; smoke data root is `./smoke_data`).
- `metadata/aptos_meta.csv`: `image_path, label (0-4), patient_id (may equal image id)`
- `metadata/aptos_splits.csv` (written by `build_splits`): above + `group, split (dev|test), fold (0-4 for dev, -1 for test)`
- `metadata/idrid_meta.csv`: `image_path, label (0-4), ma_mask, he_mask, ex_mask, se_mask, has_masks (0|1)`;
  mask columns are a relative path or an empty string (empty = lesion type absent/unannotated for that image).
- `metadata/ddr_meta.csv`: `image_path, label (0-4), gradable (0|1; 1 for all since no class 5), split (train|test)`
- `metadata/*_cache.csv` (written by `build_cache`): the table above + `cache_path, box_x0, box_y0, box_x1, box_y1, pad_top, pad_left, side, size`
Rules: deterministic row order; assert files exist; assert label ranges; log class counts per split.

## 4. Folder layout
```
data/                 # real data (gitignored)      smoke_data/   # synthetic (gitignored)
metadata/             # canonical CSVs (small; commit them)
cache/<size>_<clahe>/ # preprocessed uint8 PNG cache (gitignored)
artifacts/            # dr_effb3_fold{0..4}.pt, quality_mnv3s.pt, calibration.json, quality_threshold.json
outputs/{predictions,tables,figures,xai_maps,calibration,uncertainty}/
```
Add `data/`, `smoke_data/`, `cache/`, `outputs/`, `artifacts/*.pt`, `*.db` to `.gitignore`.

## 5. Checkpoint format
`torch.save({"state_dict": sd, "meta": {"arch": "efficientnet_b3", "num_classes": 5, "img_size": 384, "mean": [...], "std": [...],
"clahe": false, "class_names": [...], "trained_on": "aptos_dev_fold0" | "synthetic_smoke", "epoch": n, "val_qwk": x,
"seed": s, "timm_version": "..."}}, path)`. The meta block is read by the API and by evaluation, so inference never depends on
config defaults that may have changed since training.
