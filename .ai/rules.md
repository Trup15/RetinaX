# Engineering & Science Rules — RetinaX (improved)

## A. Science rules (from the blueprint; violating any of these invalidates results)
1. **Split discipline.** APTOS → {internal test (held out once), 5 dev folds}. IDRiD and DDR-test are external:
   never used to train, early-stop, select a checkpoint, fit temperature, choose a referral/XAI threshold, or pick a fold.
2. **DDR is used in two separate roles — keep them separate:**
   - DDR `train`/`valid` lists → train/select the **quality model only**.
   - DDR `test` list → external evaluation of the DR classifier **and** of the full gated pipeline.
   - DDR is never used to train, calibrate or threshold the **DR classifier**.
3. **One model is declared "primary" before any external number is seen** (default: the fold-0 model; configurable in
   `configs/default.yaml: primary_fold`). External results are reported for it; the other 4 fold models are used only for
   mean ± SD and the optional ensemble.
4. **Calibration and thresholds come from the APTOS validation fold of the primary model only.** Saved to
   `artifacts/calibration.json`. Loaded, never recomputed, at inference.
5. **Preprocessing is one function** (`retinax.preprocessing.preprocess`) used for train, val, test, external, XAI and API.
6. **Report, don't promise.** Every number in a table/figure/UI comes from a file under `outputs/` written by a script.
   Targets in the PRD are hypotheses. No fabricated or "typical" numbers in code, UI or docs.
7. **QWK, balanced accuracy, macro-F1 and per-class recall** are always reported with accuracy. Rare classes (3, 4) matter.
8. **Lesion overlap needs ground truth.** It is computed only for datasets with masks (IDRiD segmentation subset).
9. Duplicates/near-duplicates must not straddle splits (perceptual-hash groups; see `ml_protocol.md §1`).
10. Claims stay cautious: "offline-capable", "designed toward rural screening". Never "clinically validated/deployed".

## B. Cold-start rule (replaces the old "heuristic fallback" rule)
The old rule let a pixel-counting heuristic output a DR stage. That is unsafe and scientifically indefensible, and on the
spec's own synthetic images it labelled every image Stage 4. New rule:
- Missing or unloadable **DR weights** → backend still starts. `/health` reports `models: {dr: false}`. `/screen` returns the
  quality-gate result (heuristic metrics are real measurements) and `classification: null`,
  `clinical_action: "MODEL_NOT_LOADED"`. Log `[WARN] DR model not loaded; grading disabled.`
- Weights whose metadata says `trained_on: "synthetic_smoke"` load fine but every response carries
  `model_provenance: "synthetic_smoke"` and the UI shows a red "DEMO MODEL — NOT FOR CLINICAL USE" banner.
- Never download weights silently at API start. Pretrained ImageNet weights are only fetched by training scripts
  (`--pretrained`, default true outside smoke mode) and the fetch failure is a clear error with the offline workaround.

## C. Python / backend standards
- Python 3.10–3.12. Type hints. Pydantic v2. FastAPI. Compute endpoints are plain `def` (not `async def`) so the
  event loop is not blocked by torch/OpenCV; I/O-only endpoints may be `async`.
- Device: `cuda if available else cpu` via one helper `retinax.utils.get_device()`; env `RETINAX_DEVICE` overrides.
- `torch.inference_mode()` for inference. `torch.load(path, map_location="cpu", weights_only=True)`; checkpoints are
  plain dicts (`state_dict` + JSON-able metadata) so this works.
- Image bytes → `cv2.imdecode`; never write uploads to disk. Reject > 15 MB and non-images with
  `HTTPException(400, {"code": "INVALID_IMAGE", ...})`. **Do not catch `HTTPException` in a blanket `except Exception`.**
- `opencv-python-headless` only. Augmentation is plain numpy/OpenCV (`reference/augment.py`); no albumentations.
- Seeds: `seed_everything(cfg.seed)` (python, numpy, torch, cuda); log seed, versions, GPU, git hash into every run's `run_info.json`.
- All paths come from config or env, resolved relative to the repo root (`Path(__file__)`-based), never the CWD.
- Everything runnable as `python -m retinax.<module> --config configs/<name>.yaml`.

## D. Frontend standards (unchanged intent)
- Light clinical theme, 3-zone top bar, tabular numerals for probabilities, no `alert()/confirm()`, no chatbots.
- If the backend is unreachable: show "Operating in Client-Side Edge Mode" and run **only** the client-side quality
  check (Laplacian variance, illumination). The client must **not** display a DR grade, heatmap, or lesion metrics
  in that mode.
- Research-summary page reads `GET /api/v1/results` (serves files from `outputs/tables/`). No numbers embedded in TS.
  If no results exist, show "No experiment results yet".

## E. What the AI must avoid
- Hard-coded T, thresholds, Dice/IoU, ECE, accuracy, latency or "selective accuracy" values.
- Returning lesion-overlap metrics from `/screen` for uploaded images.
- `localStorage` secrets, patient data written outside the DB, cloud inference calls in the screening path.
- Deprecated torch idioms. `pickle`-based loading of untrusted files.
