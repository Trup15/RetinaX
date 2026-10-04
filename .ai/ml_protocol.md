# ML Protocol — exact recipes for training, calibration, uncertainty, XAI (NEW FILE)

Purpose: remove guesswork so the first real training run gives a sensible model. Every default lives in
`configs/default.yaml`; `configs/smoke.yaml` overrides only size/epochs/pretrained for the CPU smoke test.
None of the numbers below are results; they are settings or sanity bands.

## 0. Canonical metadata (decouples messy downloads from the pipeline)
All code reads CSVs in `metadata/`, never raw dataset folders. `prepare_aptos.py / prepare_idrid.py / prepare_ddr.py`
(written by the agent after inspecting the real download) produce them. Columns are fixed in `data_spec.md §3`.
`reference/make_synthetic_data.py` writes the same CSVs, so every later script is testable with no downloads.

## 1. APTOS splits (leakage-safe)
1. Compute a 16×16 gray difference-hash per image. Union images whose Hamming distance ≤ 4 into a `group`
   (APTOS has near-duplicates and no patient id; this is the proxy for the blueprint's patient-level rule).
2. `StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)` on (`label`, `group`): fold 0 → **internal test**
   (~1/6, frozen forever, touched only by `evaluate_dr`).
3. Remaining ~5/6 → `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` → `fold` ∈ 0..4 (dev folds).
4. Save to `metadata/aptos_splits.csv` (`split` ∈ {dev,test}, `fold`, `group`). Assert no group appears in two splits and
   every class appears in every fold (class 3/4 are small: ≥ 1 per fold is the hard minimum, otherwise reduce folds and log it).

## 2. Preprocessing + cache (big speed and correctness win)
- `preprocess()` = FOV box (Otsu with safe fallbacks) → centred square pad → resize to `img_size` → optional CLAHE on LAB-L.
  Never crop to nothing; never raise (see `reference/fundus_preproc.py`, tested).
- Build a one-time uint8 cache `cache/<img_size>_<clahe>/<id>.png` plus `metadata/*_cache.csv` with the crop params
  (`box`, `pad_top`, `pad_left`, `side`). Training reads the cache (raw APTOS images are up to ~4000×3000 px; decoding them
  every epoch is the usual training bottleneck). Cache key includes `img_size` and `clahe`, so ablations don't collide.
- IDRiD masks are transformed with the **same stored params** (`apply_params_to_mask`, nearest-neighbour). Heatmaps are
  computed in model space, so masks and heatmaps live on the same `img_size × img_size` grid. Never resize a heatmap to
  the original 4288×2848 frame; it is slower and adds an interpolation error that distorts tiny-lesion Dice.

## 3. DR classifier (EfficientNet-B3) — default recipe
| Item | Default | Why |
|---|---|---|
| Model | `timm.create_model("efficientnet_b3", pretrained=True, num_classes=5, drop_rate=0.3, drop_path_rate=0.2)` | ImageNet transfer; head dropout is also what MC-Dropout uses |
| Input | 384 px (config `img_size`); 512 px for the final run if GPU memory allows | microaneurysms are tiny; 300 px (native) loses them |
| Normalisation | ImageNet mean/std, stored in checkpoint metadata | train/infer parity |
| Loss | CE with class weights ∝ 1/sqrt(freq) (config `loss: ce_sqrtw`) | stronger weights hurt calibration; ablation covers `ce`, `ce_sqrtw`, `focal` |
| Optimiser | AdamW, lr 2e-4, wd 1e-4, 1 warm-up epoch then cosine to 1e-6, grad-clip 1.0 | stable for EfficientNet fine-tuning |
| Epochs | 25 (+ early stop patience 8 on val QWK) | |
| Batch / AMP | 16 (24 GB GPU: 32); `torch.autocast` on CUDA only | |
| Augmentation | `reference/augment.py` (flips, ±20°, 0.92–1.08 scale, small shift, contrast/brightness, mild gamma) | lesion-safe |
| Checkpoint rule | best val **QWK**, tie → lower val loss. Save `{state_dict, meta}` | QWK is the ordinal metric |
| Logs | per-epoch JSON line: loss, acc, balanced acc, macro-F1, QWK, ECE | |
At the end of every fold run save `outputs/predictions/aptos_fold{f}_val.npz` (`ids, labels, logits`) — **calibration reads only this**.
No label smoothing, no mixup (they distort the probabilities that calibration/uncertainty later depend on).

**Sanity bands (bug detectors, NOT targets and NOT claims):** after a proper 384 px run, APTOS internal QWK well below
~0.80 or accuracy below ~70 % almost always means a bug (label mapping, double normalisation, BGR/RGB mix-up, cache built
with a different size, LR too high, wrong class order). Fix the bug before tuning anything. Never quote these bands in the paper.

Cheap accuracy levers, in priority order: (1) 384→512 px, (2) pretrained weights actually loaded (log the missing/unexpected
keys), (3) 5-fold ensemble average of logits (free, models already trained), (4) CLAHE ablation, (5) loss ablation.
Do **not** switch to regression/ordinal heads for the primary model: the blueprint needs softmax probabilities for entropy.
(A regression head may be added later as an explicitly labelled comparison.)

## 4. Baselines (blueprint Phase 3) — keep tiny
Same recipe, same folds, fold 0 only: `resnet50`, `efficientnet_b0`, `mobilenetv3_large_100` (via timm). One table. Their only purpose
is to show EfficientNet-B3 is a reasonable choice. ViT/Swin optional and last.

## 5. Quality model (MobileNetV3-Small on DDR)
- Label: `ungradable = 1 if DDR class == 5 else 0` (model output index 1 = ungradable; **document and test the index mapping**).
- Train on DDR `train` list, select on `valid`, report on `test` (ungradable class: sensitivity, specificity, precision, F1, AUROC, confusion matrix).
- Imbalance: ungradable is the minority. Use weighted CE (weight ∝ 1/freq) or a `WeightedRandomSampler`; report both classes.
- `timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=2)`, input 320 px (blur needs resolution), AdamW lr 1e-3 → cosine, 15 epochs.
- Operating point: choose the probability threshold on **valid** so ungradable sensitivity ≥ `quality.target_sensitivity` (default 0.95);
  store in `artifacts/quality_threshold.json`; report the specificity cost on test.
- DDR only has a binary-ish quality label; blur vs cataract vs illumination **cannot** be learned from it. Report the Laplacian
  variance / illumination / FOV-coverage numbers as "diagnostic hints", never as a classified failure cause.
- Gate pass-rate on APTOS and IDRiD is reported as information only (no quality labels there).

## 6. External evaluation (unchanged model, no fine-tuning)
For the primary model: APTOS internal test, IDRiD (all 516 graded images), DDR test (grades 0–4 only; count and report class-5 images separately).
Metrics + 95 % bootstrap CI (1000 resamples): accuracy, balanced accuracy, macro-F1, QWK, per-class recall/precision/F1,
macro one-vs-rest AUROC, referable (grade ≥ 2) sensitivity/specificity/AUROC, confusion matrix, NLL, Brier, ECE (before/after T),
mean normalised entropy. Save raw `ids, labels, logits` for each dataset to `outputs/predictions/<dataset>_primary.npz`.
Pipeline variant: DDR-test with the quality gate in front, retained subset only, plus the gate rejection rate.
DDR grade distribution differs from APTOS; do not "fix" it by re-weighting — it is part of the domain shift being measured.

## 7. Calibration
- `fit_temperature(val_logits, val_labels)` on `aptos_fold{primary}_val.npz` (NLL, single scalar T, `reference/uncertainty_metrics.py`).
- Report ECE (15 bins), Brier, NLL before/after on val, internal test, IDRiD, DDR-test. Reliability diagrams for each.
- Save `artifacts/calibration.json`: `{temperature, fitted_on, n_val, ece_before, ece_after, seed}` — the file is the single source of T.
- T is a result of fitting. Nothing in code or docs may contain a literal temperature value.

## 8. Uncertainty and referral
Compute for every dataset, from calibrated probabilities:
1. **MSP**: `u = 1 − max p`.
2. **Predictive entropy** (natural log, divided by ln 5 so u ∈ [0,1]) — `predictive_entropy`.
3. **MC-Dropout** (N = 30 by default, `mc.n_passes`): head-only dropout. Compute backbone features once, then re-apply the
   dropout+classifier N times (cheap: backbone cost × 1). `model.eval()`, then use `F.dropout(feats, p=drop_rate, training=True)` before
   the classifier. Uncertainty = entropy of the mean probability; also report mutual information. Verify in the smoke test that
   repeated passes **differ** (`std > 0`) and that BatchNorm stayed in eval mode.
   Limitation to state honestly: head-only dropout gives limited diversity; the 5-fold ensemble (item 4) is the stronger comparison.
4. **Fold ensemble (optional, free)**: average calibrated probabilities of the 5 fold models; entropy / mutual information.
Evaluate each: AUROC of `u` for detecting errors (correct vs incorrect), AURC and risk–coverage curve, entropy distributions for
correct vs incorrect and internal vs external (Mann–Whitney U, effect size).
**Referral threshold τ** is chosen on the APTOS **validation** fold for a pre-declared referral rate (`referral.target_rate`, default 0.15),
saved per method in `artifacts/calibration.json`, and applied unchanged to every dataset. Report for each dataset: referral rate,
selective accuracy, selective QWK, error-capture rate, and the full accuracy-vs-referral-rate table (0/5/10/20/30 %).
Also report clinically weighted errors: share of retained errors that are ≥ 2 grades off, and retained referable-DR sensitivity.
Expectation (hypothesis H4, to be tested not assumed): on shifted data the same τ refers *more* cases.
The PRD's "> 95 % selective accuracy" is a hypothesis, not a requirement.

## 9. XAI + lesion validation
Targets: IDRiD segmentation-subset images that have ≥ 1 non-empty lesion mask. Explained class = predicted class (primary) and ground-truth
class (secondary). All maps are in model space (`img_size²`), multiplied by the FOV disc mask (`fov_circle_mask`) and normalised to [0,1].
| Method | Library | Settings |
|---|---|---|
| Grad-CAM++ | `pytorch-grad-cam` `GradCAMPlusPlus` | target layer `model.blocks[-1]` (384 ch for B3). Print the layer's output shape in the smoke test. |
| Score-CAM | `pytorch-grad-cam` `ScoreCAM` | same layer, `batch_size=32`; slow on CPU → run only on the lesion subset |
| Integrated Gradients | `captum.attr.IntegratedGradients` | baseline = black image (zeros in **raw** pixel space, i.e. `(0−mean)/std` after normalisation), `n_steps=32`, `internal_batch_size=8`; map = Σ_c |attr|, Gaussian blur σ=4 |
Resolution caveat: the last block has stride 32 (12×12 at 384 px), so CAM maps are coarse compared with microaneurysms (a few pixels).
Expect low absolute Dice/IoU for small lesions; this is a finding, not a bug. Optional extra: CAM from `blocks[-3]` (stride 16) as a labelled variant.
**Binarisation (all fixed in config before looking at any IDRiD result):**
(a) fixed thresholds {0.25, 0.5, 0.75}; (b) top-fraction masks {2, 5, 10, 20 %} of FOV pixels; (c) threshold-free: pointing game (tol 0 and tol = `img_size/24` px)
and pixel-wise ROC-AUC of the heatmap against the lesion mask.
**Metrics:** Dice, IoU, precision, recall, pointing game, lesion-AUROC — per lesion type (MA/HE/EX/SE) and for the union mask; deletion/insertion AUC (20 steps, blur-fill)
as the faithfulness check (optional but listed in the blueprint).
**Context baselines (essential for interpretation):** random mask of equal area (`random_baseline_dice`), a centre-Gaussian map, and an input-only
"dark-lesion" map from the inverted green channel. A method only counts as lesion-aware if it clearly beats these.
**Statistics:** paired Wilcoxon signed-rank between methods per metric, Holm correction, bootstrap 95 % CIs over images (N≈81 → wide CIs; say so).
Also answer blueprint §20: correct vs misclassified images, confidence vs overlap (Spearman), severe vs mild.

## 10. Deployment benchmark (measured, never asserted)
`retinax.deploy.benchmark`: parameter count, fp32 checkpoint size, ONNX export (opset 17) + `onnxruntime` parity (max |Δlogit| < 1e-3 on 20 images, else report the
difference), CPU latency (batch 1, threads 1 and 4, 10 warm-up + 50 timed runs, median and p95) at the configured `img_size`, peak RSS via `psutil`,
GPU latency if CUDA is present. INT8 quantisation attempt wrapped in `try/except` and reported as "not attempted/failed" if it fails.
Output `outputs/tables/deploy_benchmark.json`. Language in any text: "offline-capable", "CPU-benchmarked". PRD targets (15 MB, 45 ms) are aspirations; B3 fp32 is tens of MB.

## 11. Ablations (each answers one question; fold 0 unless stated)
CLAHE vs none · CE vs weighted CE vs focal · no calibration vs T · MSP vs entropy vs MC-Dropout vs ensemble · XAI method vs method · gate on vs off (DDR-test) ·
XAI threshold sensitivity · internal vs external. Nothing else.

## 12. Reproducibility record
Every script writes `outputs/<run>/run_info.json`: seed, python/torch/timm/CUDA versions, GPU name, config snapshot, git hash, dataset CSV checksums,
selected checkpoint epoch/metric. Predictions, logits, uncertainties, XAI maps (`.npz`/PNG) and per-image overlap CSVs are saved for every table.
