# System Architecture — RetinaX (improved)

## 1. Design principle: library first, apps second
One installable Python package, `retinax` (under `src/`, `pip install -e .`), holds **all** ML logic. The experiment scripts and the FastAPI
backend are thin wrappers that import it. This guarantees training, evaluation, XAI and the API share the same preprocessing,
model loading, calibration and uncertainty code (the old layout duplicated logic inside `main.py`).

```
CLIENT (React 19 + TS + Vite + Tailwind)  ──HTTP/JSON──▶  BACKEND (FastAPI)  ──imports──▶  retinax (PyTorch, OpenCV, timm)
                                                              │                                   │
                                                              ▼                                   ▼
                                                       SQLite (referrals)               artifacts/ (weights, calibration.json)
```

## 2. Repository layout
```
retinax-root/
├── .ai/                      # these spec files
├── docs/diabetic_retinopathy_project_blueprint.md
├── pyproject.toml            # package "retinax" (src layout), extras: [dev], [xai], [deploy]
├── requirements.txt          # see templates/requirements.txt
├── configs/default.yaml, smoke.yaml
├── metadata/                 # canonical CSVs
├── src/retinax/
│   ├── config.py             # load YAML + env overrides -> dataclass; resolves paths from repo root
│   ├── utils.py              # get_device, seed_everything, run_info, logging
│   ├── data/                 # prepare_{aptos,idrid,ddr}.py, build_splits.py, build_cache.py, datasets.py
│   ├── preprocessing/        # preprocess.py (from reference/fundus_preproc.py), augment.py
│   ├── models/               # factory.py (timm create), io.py (save/load ckpt+meta)
│   ├── train/                # train_dr.py, train_quality.py, losses.py
│   ├── eval/                 # evaluate_dr.py, metrics.py (QWK, F1, AUROC, bootstrap), quality_eval.py
│   ├── uncertainty/          # calibrate.py, metrics.py (from reference), mc_dropout.py, analyze.py, referral.py
│   ├── xai/                  # cams.py (GradCAM++/ScoreCAM), ig.py, baselines.py, lesion_metrics.py (from reference), run_lesion_validation.py
│   ├── deploy/               # benchmark.py, export_onnx.py
│   └── pipeline.py           # RetinaXPipeline: quality gate -> DR -> calibrated uncertainty -> referral -> optional XAI
├── backend/
│   ├── __init__.py
│   └── app/
│       ├── __init__.py
│       ├── main.py           # FastAPI app, lifespan loads RetinaXPipeline once
│       ├── api/v1/{screening.py, referrals.py, results.py, cohorts.py}
│       ├── schemas.py        # Pydantic v2
│       └── db.py             # SQLAlchemy 2.0 (SQLite default)
├── frontend/                 # Vite React TS app (see design.md); may need to be created
├── scripts/                  # make_synthetic_data.py, run_smoke_test.py
├── tests/                    # pytest (see tasks.md)
├── artifacts/  outputs/  cache/  data/  smoke_data/   # gitignored where large
```
Import paths: ML code `from retinax...`; API run from repo root: `python -m uvicorn backend.app.main:app --port 8000`.
(The old `cd backend && python -m app.main` / `uvicorn.run("main:app")` combination fails with an import error.)

## 3. Command-line interface (every command: `python -m <module> --config configs/<name>.yaml [flags]`)
| # | Module | Reads | Writes |
|---|---|---|---|
| 1 | `retinax.data.build_splits` | `metadata/aptos_meta.csv` | `metadata/aptos_splits.csv` |
| 2 | `retinax.data.build_cache` | all meta CSVs | `cache/...`, `metadata/*_cache.csv` |
| 3 | `retinax.train.train_dr --fold N` | cache + splits | `artifacts/dr_effb3_fold{N}.pt`, `outputs/predictions/aptos_fold{N}_val.npz` |
| 4 | `retinax.train.train_quality` | `ddr_cache.csv` | `artifacts/quality_mnv3s.pt`, `outputs/tables/quality_*.json` |
| 5 | `retinax.eval.evaluate_dr --dataset {aptos_test,idrid,ddr_test}` | primary ckpt | `outputs/predictions/<ds>_primary.npz`, `outputs/tables/dr_<ds>.json` |
| 6 | `retinax.uncertainty.calibrate` | `aptos_fold{primary}_val.npz` | `artifacts/calibration.json`, `outputs/calibration/*` |
| 7 | `retinax.uncertainty.analyze` | predictions + calibration | `outputs/tables/uncertainty_*.json`, `outputs/uncertainty/*`, risk-coverage CSV/PNG |
| 8 | `retinax.xai.run_lesion_validation` | primary ckpt + IDRiD masks | `outputs/tables/xai_lesion.csv`, `outputs/xai_maps/*`, summary JSON |
| 9 | `retinax.deploy.benchmark` | checkpoints | `outputs/tables/deploy_benchmark.json` |
| 10 | `retinax.pipeline --image PATH` | artifacts | JSON to stdout (same schema as the API) |
| 11 | `retinax.results_summary` | `outputs/tables/*` | `outputs/tables/summary.json` (served by `/api/v1/results`, rendered by the UI) |
Every command returns exit code 0 on success, non-zero with a one-line cause on failure, and writes `run_info.json`.

## 4. Pipeline logic (`RetinaXPipeline.screen(image_bgr, xai_method=None)`)
```
decode → preprocess(img_size) ──▶ quality model (320 px, own preprocess of same crop) → p_ungradable
   p_ungradable ≥ quality threshold? ── yes ──▶ {status: UNGRADABLE, action: RECAPTURE_IMAGE, classification: null}
   no ──▶ DR model → logits → calibrated probs (T from calibration.json) → entropy_norm (+ MC-Dropout if enabled)
        entropy ≥ tau (from calibration.json)? ── yes ──▶ action: SPECIALIST_REFERRAL (grade still returned as "suggested")
        no ──▶ action: ACCEPT_GRADE
   optional: heatmap (Grad-CAM++ default) overlay as base64 PNG
```
Notes: the quality model and the DR model may use different input sizes; both consume the *same* FOV crop (crop once, resize twice).
Quality threshold, T and τ are loaded from `artifacts/`; if a file is missing the corresponding step reports `"unavailable"` instead of using a default.

## 5. Persistence (SQLite by default)
Tables: `patients(id, external_id, age, sex)`, `screenings(id, patient_id, eye, quality_status, predicted_stage, entropy_norm, action, model_provenance, created_at)`,
`referrals(id, screening_id, reason, status, created_at)`. **No image bytes in the DB.** Heatmaps are returned inline and not stored by default.
Create tables at startup (`Base.metadata.create_all`) so a fresh checkout works; no migrations in v1.
(`lesion_annotations` and `model_checkpoints` tables from the old spec are dropped: annotations live in the IDRiD files, checkpoint info lives in the checkpoint meta.)

## 6. Security / privacy (kept minimal and honest)
Local-only by default (`127.0.0.1`), CORS limited to the Vite dev origins (`http://localhost:5173`, `http://127.0.0.1:5173`), upload size limit, no auth in v1
(state this in the README: not for networked clinical use). No telemetry, no external calls in the screening path.
