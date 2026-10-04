# Project Memory & Decision Records — RetinaX (improved)

Rule for this file: record **decisions and reasons**, never experimental outcomes that have not been measured.
Measured outcomes live in `outputs/` and the paper, not here.

## ADR-01 Dual-gated pipeline (kept)
Quality gate (MobileNetV3-Small) → DR classifier (EfficientNet-B3) → uncertainty gate. Reason: grading ungradable images risks false reassurance;
the blueprint's central framing is a trustworthy *pipeline*, not a bigger classifier.

## ADR-02 Backbones (kept, claims softened)
MobileNetV3-Small (small, fast) for quality; EfficientNet-B3 for grading (strong transfer-learning baseline, manageable cost, easy XAI hooks).
Parameter counts / sizes are **measured** by `deploy.benchmark`, not asserted.

## ADR-03 Calibration (changed)
Single-scalar temperature scaling, fitted by NLL on the APTOS validation fold of the primary model, stored in `artifacts/calibration.json`.
*Old text claimed T = 1.38 and specific ECE reductions before any model existed — removed.* The value of T is an output of fitting.

## ADR-04 XAI benchmark (changed)
Grad-CAM++ is the **default display** method for speed only. Which method best matches lesions is a research question answered by
`run_lesion_validation`. *Old text declared a ranking and Dice values before running anything — removed.* Expect coarse CAM resolution (stride 32) to limit small-lesion overlap; baselines (random/centre/inverted-green) give the context.

## ADR-05 Client-side fallback (changed)
Frontend fallback is a **quality-only** edge mode (Laplacian variance, illumination). It never shows a grade, heatmap, or lesion metric. *Old fallback simulated staging and Dice in the browser, which could be mistaken for model output.*

## ADR-06 Library-first architecture (new)
All ML logic in package `retinax`; CLI scripts and FastAPI are thin wrappers. Reason: one preprocessing/loading/calibration implementation for training, evaluation, XAI and API.

## ADR-07 Canonical metadata CSVs (new)
Pipelines read CSVs, not raw dataset folders. Reason: Kaggle mirrors vary; synthetic smoke data can emit identical CSVs; paths stay out of code.

## ADR-08 DDR dual role (new, resolves a contradiction)
DDR train/valid → quality model only. DDR test → external evaluation. DDR never trains/calibrates/thresholds the DR classifier. (Old rules said "DDR never used for training" while also training the quality model on it.)

## ADR-09 Lesion overlap only with ground truth (new)
Overlap metrics need a lesion mask. They are computed for IDRiD segmentation-subset images and never returned for arbitrary uploads. (Old `/screen` returned constant Dice/IoU for any image.)

## ADR-10 Preprocess once, cache (new)
Deterministic FOV-crop + square-pad + resize cached as PNG per (size, CLAHE). Masks are transformed with the stored crop parameters; heatmaps stay in model space.

## ADR-11 No heuristic DR staging (new)
A pixel-count rule produced "Stage 4" for every synthetic image (including the Normal one). Without a model the API returns `MODEL_NOT_LOADED`; with the smoke model it is labelled `synthetic_smoke`.

## ADR-12 Plain numpy/OpenCV augmentation (new)
Avoids albumentations API changes between 1.x and 2.x. Behaviour is specified in `reference/augment.py`.

## ADR-13 Primary model declared in advance (new)
`primary_fold` is fixed in config before external evaluation; other folds give mean ± SD and an optional ensemble. Prevents picking the fold that looks best on IDRiD/DDR.

## Key constants (configuration, not facts)
Image sizes: DR 384 (512 for final), quality 320, smoke 160. Referral target rate 0.15 (pre-declared, editable). Quality target sensitivity 0.95. XAI thresholds/fractions in `configs/default.yaml`.
There are deliberately **no** constants for T, τ, blur threshold or Dice: they come from fitted files.

## Deviations log (agent appends here)
- (empty)
