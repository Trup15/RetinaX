# Implementation Roadmap & Verifiable Tasks — RetinaX (improved)

Rule: do not start a task until the previous task's **Check** passes. Every check is a command with an observable result.
Phases 0–7 = the first build (CPU-only, no downloads). Phase 8 = real data on a GPU (needs the human). Phase 9 = UI.
Shell examples are bash; on Windows run the same `python -m ...` commands in PowerShell.

## Phase 0 — Scaffold
- [ ] **0.1 Package skeleton.** Create `pyproject.toml` (src layout, package `retinax`, console deps from `templates/requirements.txt`), the folder layout in `architecture.md §2`, `.gitignore` (`templates/gitignore.txt` (rename to `.gitignore`)).
  *Check:* `python -m venv .venv && . .venv/bin/activate && pip install -e . && python -c "import retinax; print('ok')"`
- [ ] **0.2 Dependencies.** Install torch for the machine first, then `pip install -r requirements.txt`.
  *Check:* `python -c "import torch, timm, cv2, fastapi, pytorch_grad_cam, captum, sklearn; print(torch.__version__, timm.__version__)"` and `python -c "import timm; m=timm.create_model('efficientnet_b3', pretrained=False, num_classes=5); print(sum(p.numel() for p in m.parameters())/1e6)"` prints a number near 10–12 (millions).
- [ ] **0.3 Reference modules in.** Copy `reference/fundus_preproc.py, augment.py, uncertainty_metrics.py, lesion_metrics.py, config_loader.py` into `src/retinax/...` (config_loader becomes `retinax/config.py`; set `REPO_ROOT` to the real repo root) (see `architecture.md`), copy `reference/test_reference.py` to `tests/` and fix its imports.
  *Check:* `pytest tests/test_reference.py -q` → all pass.
- [ ] **0.4 Config + utils.** Copy `templates/configs/*.yaml` to `configs/`; `retinax.config.load_config` (from `reference/config_loader.py`: handles `_extends` deep-merge, env overrides, repo-root paths), plus `get_device`, `seed_everything`, `run_info`.
  *Check:* `python -c "from retinax.config import load_config; c=load_config('configs/smoke.yaml'); print(c.dr.img_size, c.dr.pretrained, c.dr.epochs)"` prints `160 False 2`.

## Phase 1 — Data layer (smoke data + real DDR)
- [ ] **1.1 Synthetic data (APTOS/IDRiD only).** Copy `reference/make_synthetic_data.py` → `scripts/`. 
  *Check:* `python scripts/make_synthetic_data.py --out smoke_data` creates 2 metadata CSVs (aptos_meta.csv, idrid_meta.csv); every `image_path` exists.
- [ ] **1.2 Real DDR metadata.** Run `python scripts/prepare_ddr.py --data-root smoke_data/DR_grading --label-file smoke_data/DR_grading.csv`.
  *Check:* Creates `metadata/ddr_meta.csv` with 12,522 rows (9,376 train + 3,146 test), labels 0–4, all gradable=1, split column populated.
- [ ] **1.3 Splits.** `retinax.data.build_splits` per `ml_protocol.md §1` (hash groups, `StratifiedGroupKFold`) — APTOS only.
  *Check:* test asserts: no `group` in two splits, test fold non-empty, `fold` ∈ 0..4 for dev rows, every class present in test+dev (on smoke data classes 3/4 may be tiny — then the check is "assertion logs a warning and reduces folds", not a crash).
- [ ] **1.4 Cache.** `retinax.data.build_cache` writes the PNG cache + `*_cache.csv` with crop params for all datasets.
  *Check:* every `cache_path` exists, shape == `(img_size, img_size, 3)`, IDRiD mask transformed with stored params has non-zero pixels where the original mask had them (test with `apply_params_to_mask`).
- [ ] **1.5 Dataset class.** `datasets.py` (`DRDataset`, `QualityDataset`) reading the cache CSVs; augmentation only in train mode; ImageNet normalisation; returns `(tensor, label, id)`.
  *Check:* a DataLoader batch has shape `(B,3,S,S)`, finite values, labels within range; two loads of an eval sample are identical, two loads of a train sample differ.

## Phase 2 — Training (smoke scale)
- [ ] **2.1 DR training.** `retinax.train.train_dr --fold 0 --config configs/smoke.yaml` (recipe in `ml_protocol.md §3`; smoke: `pretrained: false`, `img_size: 160`, `epochs: 2`, `batch_size: 8`).
  *Check:* exits 0; creates `artifacts/dr_effb3_fold0.pt` whose meta has `trained_on == "synthetic_smoke"`, `outputs/predictions/aptos_fold0_val.npz`, a per-epoch JSON log; loss is finite; `torch.load(..., weights_only=True)` succeeds.
- [ ] **2.2 Quality training.** `retinax.train.train_quality` (label mapping: index 1 = ungradable).
  *Check:* exits 0; `artifacts/quality_mnv3s.pt`; `quality_threshold.json` written from the **valid** split; test asserts a model fed a sharp vs a heavily blurred version of the same synthetic image gives `p_ungradable(blurred) > p_ungradable(sharp)` after smoke training (or, if the 2-epoch smoke model is too weak, assert the label-index mapping with a hand-built logits example).
- [ ] **2.3 Resume + determinism.** `--resume` works; two runs with the same seed give the same first-epoch loss on CPU to 1e-5.

## Phase 3 — Evaluation, calibration, uncertainty
- [ ] **3.1 Metrics.** `eval/metrics.py`: accuracy, balanced accuracy, macro-F1, per-class P/R/F1, QWK, OvR AUROC (guard classes absent from a split), referable binary metrics, bootstrap CIs.
  *Check:* unit tests on tiny hand-made arrays (QWK of identical vectors = 1; known confusion matrix → known F1).
- [ ] **3.2 Evaluate datasets.** `retinax.eval.evaluate_dr --dataset aptos_test|idrid|ddr_test`.
  *Check:* writes `outputs/predictions/<ds>_primary.npz` + `outputs/tables/dr_<ds>.json`; the JSON contains every metric from `ml_protocol.md §6` (values may be `null` where undefined, never missing).
- [ ] **3.3 Calibration.** `retinax.uncertainty.calibrate` fits T on `aptos_fold0_val.npz` only.
  *Check:* `artifacts/calibration.json` has `temperature > 0`, `fitted_on` mentions the val fold; argmax unchanged after scaling (test); `grep -R "1\.38" src backend frontend` finds nothing.
- [ ] **3.4 Uncertainty + referral.** `retinax.uncertainty.analyze` (MSP, entropy, MC-Dropout, ensemble if >1 fold model exists).
  *Check:* MC passes differ (`std>0`); referral rate on the **validation** set at τ is within ±2 pp of the target rate; risk–coverage CSV ends at coverage 1.0; all outputs finite.

## Phase 4 — XAI + lesion validation
- [ ] **4.1 CAMs.** `xai/cams.py` Grad-CAM++ and Score-CAM on `model.blocks[-1]`; `xai/ig.py` Integrated Gradients.
  *Check:* each returns a float map of shape `(S,S)` in `[0,1]`, not constant, finite; log the target-layer output shape; wrapper restores the model to eval mode and removes hooks.
- [ ] **4.2 Lesion metrics.** wire `lesion_metrics.py` (tested) + baselines (random, centre-Gaussian, inverted-green).
  *Check:* test with a heatmap equal to the mask → Dice 1.0, pointing game True; random baseline Dice ≈ lesion fraction-level (low).
- [ ] **4.3 Run validation.** `retinax.xai.run_lesion_validation` over IDRiD images having masks, all three methods + baselines, all binarisations in config.
  *Check:* `outputs/tables/xai_lesion.csv` has one row per (image, method, binarisation, lesion type); `IoU == Dice/(2-Dice)` for every row (±1e-9); Wilcoxon/Holm table and bootstrap CI JSON written; thresholds read from config (grep shows no literals in the script).

## Phase 5 — Pipeline + deployment benchmark
- [ ] **5.1 Pipeline class + CLI.** `retinax.pipeline --image smoke_data/idrid/images/IDRiD_000.png` prints JSON matching the schema in `backend_integration.md §2`.
  *Check:* probabilities sum to 1 ± 1e-6; blurred image → `UNGRADABLE` path returns `classification: null`; with `artifacts/` renamed away → `MODEL_NOT_LOADED` and exit code 0.
- [ ] **5.2 Benchmark.** `retinax.deploy.benchmark` writes `outputs/tables/deploy_benchmark.json` (params, size, latency median/p95 for 1 and 4 threads, peak RSS, ONNX parity or a recorded reason it failed).

## Phase 6 — API
- [ ] **6.1 Backend.** Implement per `backend_integration.md` (lifespan load, routers, SQLite).
  *Check:* `python -m uvicorn backend.app.main:app --port 8000 &` then `curl -s localhost:8000/api/v1/health` returns JSON with `"status":"healthy"` (note: JSON spacing — parse it, don't `grep` an exact string with/without a space).
- [ ] **6.2 API tests.** `pytest tests/test_api.py -q` (list in `backend_integration.md §6`) passes, including the **no-artifacts** and **invalid-image → 400** cases.
- [ ] **6.3 Curl check.** `curl -s -X POST localhost:8000/api/v1/screen -F "file=@smoke_data/idrid/images/IDRiD_000.png" -F "eye=OD"` returns keys `quality, classification, uncertainty, clinical_action`.

## Phase 7 — Smoke test gate (the definition of "first build done")
- [ ] **7.1** `python scripts/run_smoke_test.py` runs Phases 1–6 commands in order on `configs/smoke.yaml`, stops at the first failure, prints a per-step PASS/FAIL table, exits 0 only if all pass. (A starting version is in `templates/scripts/run_smoke_test.py`; it has not been executed by the spec author — the agent runs it and fixes it.)
- [ ] **7.2** Full `pytest -q` is green. Record versions and wall-clock in `memory.md`.

## Phase 8 — Real data (human prerequisites in README_FIRST)
- [ ] **8.1 Prepare APTOS/IDRiD.** Inspect each real download, write `prepare_aptos.py / prepare_idrid.py`, produce canonical CSVs; print class counts and image-size stats; stop and ask on any missing file.
  **DDR is already prepared** — 12,524 images (9,378 train + 3,146 test) in `smoke_data/DR_grading/` with `metadata/ddr_meta.csv`.
- [ ] **8.2 Train** (GPU): `train_dr --fold 0` first (`img_size: 384`), check against the sanity bands in `ml_protocol.md §3`, then folds 1–4; then `train_quality`. Training on Colab/Kaggle is fine: copy back `artifacts/*.pt` and `outputs/`.
  **Note:** Quality model needs ungradable (class 5) images — generate synthetic or use alternative dataset.
- [ ] **8.3 Evaluate → calibrate → uncertainty → XAI → benchmark**, in that order (`architecture.md §3`), with the primary fold declared **before** step 8.3 starts.
- [ ] **8.4 Summary.** `retinax.results_summary` → `outputs/tables/summary.json`; paper tables/figures are generated from `outputs/` only.

## Phase 9 — Frontend (see design.md)
- [ ] **9.1** If `frontend/` is missing, create it with Vite + React + TS + Tailwind; if it exists, only add/repair `src/utils/apiClient.ts` (base URL from `import.meta.env.VITE_API_URL`, default `http://localhost:8000`).
  *Check:* `cd frontend && npm install && npm run build` exits 0.
- [ ] **9.2** Screener view calls `/screen`; shows the DEMO-MODEL banner when `model_provenance == "synthetic_smoke"`; renders `MODEL_NOT_LOADED`, `UNGRADABLE`, `SPECIALIST_REFERRAL`, `ACCEPT_GRADE` states.
- [ ] **9.3 Offline fallback.** Stop the backend, reload: UI shows "Operating in Client-Side Edge Mode", runs the client quality check only, and shows **no** grade/heatmap/overlap; no white screen.
- [ ] **9.4 Results view** reads `/api/v1/results`; with no summary shows "No experiment results yet"; contains no numeric constants from experiments in source (`grep` for `0.534`, `97.8`, `1.38`, `42 ms` finds nothing).
