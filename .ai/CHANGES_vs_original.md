# What changed vs the original `.ai/` bundle, and why

Scope kept: research design, models, datasets, split philosophy, metrics, UI visual system. Scope changed: how precisely things are specified, and
every place where the original would fail to build, fail a check, or contradict the blueprint's own "never claim unmeasured results" rule.

## A. Things that would have broken the build or the checks
| # | Original problem | Fix (where) |
|---|---|---|
| 1 | `python -m app.main` + `uvicorn.run("main:app")` cannot import from `backend/` | Run from repo root: `python -m uvicorn backend.app.main:app`; `__init__.py` files specified (architecture §2, backend §3) |
| 2 | Heuristic staging (red-pixel counts) returned Stage 4 for every synthetic image incl. "normal" (I executed the old generator + old rules) | Removed. No heuristic staging; `MODEL_NOT_LOADED` state (rules §B, ADR-11) |
| 3 | `raise HTTPException(400)` inside `try/except Exception` became a 500 | Pattern with `except HTTPException: raise` + test (backend §5–6) |
| 4 | `pytest tests/` with no tests; pytest/httpx not in requirements | Test list per phase; deps added (tasks, backend §1) |
| 5 | Task 2.1 required >90 % margin removal but a bounding-box crop leaves ~21 % black corners; said Otsu but used a fixed threshold | Tested crop + square pad + FOV disc mask; criteria rewritten (reference/fundus_preproc.py, tests) |
| 6 | Real-model path never wired; `pretrained=False` loader contradicted the cold-start rule | `RetinaXPipeline.load` with checkpoint meta; no silent downloads (backend §3–4, rules §B) |
| 7 | Quality-model output index likely reversed (DDR: ungradable = class 5) | Index 1 = ungradable, stated + tested (ml_protocol §5, tasks 2.2) |
| 8 | Weight file names/paths inconsistent (`.pt` vs `.pth`, three names) | One name per artifact, resolved from config (architecture §3, data_spec §4–5) |
| 9 | `numpy<2.0` pin, missing deps (pandas, captum, grad-cam), unused deps | New `requirements.txt` (backend §1) |
| 10 | Task 1.2 check grepped `"status":"healthy"` (JSON spacing) | Check parses JSON (tasks 6.1) |
| 11 | Frontend described as implemented but not included in the handoff | Treated as possibly missing; Phase 9 creates/wires it (current_state, tasks) |
| 12 | Heavy ML in `async def` endpoints blocks the event loop | Plain `def` compute endpoints (rules §C) |

## B. Things that contradicted the blueprint (scientific validity)
| # | Original | Fix |
|---|---|---|
| 13 | Fixed T = 1.38, ECE 8.9→3.8 %, Dice 0.534, 86 % pointing, "97.8 % selective accuracy", 15 MB / 42 ms stated as facts or hard-coded in API/UI | Removed or marked as hypotheses; T, τ, thresholds loaded from fitted files; UI reads `outputs/` (rules §A6, memory ADRs, prd §5) |
| 14 | `/screen` returned lesion overlap for any upload (no mask exists) | Lesion overlap only for IDRiD images with masks (ADR-09, backend §2) |
| 15 | "Heatmap" = red-minus-green blur labelled Grad-CAM++ | Real Grad-CAM++/Score-CAM/IG via libraries with specified layer/settings (ml_protocol §9) |
| 16 | Fixed entropy cut-off 0.85 bits refused nearly every case on the spec's own logits | τ chosen on APTOS validation for a pre-declared referral rate (ml_protocol §8) |
| 17 | Rule "DDR never used for training" vs "quality model trained on DDR" | DDR dual-role rule (rules §A2, ADR-08) |
| 18 | Calibration "fitted on APTOS validation fold" with no fold/test design | Frozen internal test + 5 dev folds, group-aware, primary fold pre-declared (ml_protocol §1, ADR-13) |
| 19 | Blur/cataract/illumination presented as classified causes | Binary quality label only; measurements shown as hints (ml_protocol §5) |
| 20 | IDRiD "516 images with masks"; APTOS "1,928 test images" implied usable | Flagged: masks only for the segmentation subset; APTOS test is unlabelled (README_FIRST, data_spec §2) |
| 21 | Dice evaluated on a 100×100 grid with unspecified mask alignment | Common model-space grid, masks transformed with stored crop params, FOV mask, area-matched binarisation, baselines, Holm-corrected tests (ml_protocol §2, §9) |

## C. Additions
`README_FIRST.md` (authority order, human-only actions, verify-first facts) · `ml_protocol.md` (training/eval/calibration/uncertainty/XAI/benchmark recipes) ·
canonical metadata CSVs and checkpoint-meta format · smoke-test gate (acceptance test for "builds and runs") · `reference/` tested code (9 passing tests) ·
`templates/` (requirements, pyproject, gitignore, configs, smoke script) · fold-ensemble uncertainty (free from the 5 fold models) · preprocessed-image cache (training speed).

## D. Honest limits of this package
- The torch/timm/FastAPI/Grad-CAM/captum snippets and `run_smoke_test.py` were **not executed** (PyTorch could not be installed in the authoring sandbox). Only the numpy/OpenCV/sklearn reference modules and config loader were run (9 passing tests). Expect small API-version fixes on first run.
- Dataset layout facts (IDRiD mask counts, DDR list files, Kaggle mirror structure) come from memory of the public descriptions; the agent must verify them on the real downloads.
- Sanity bands for accuracy are bug detectors, not targets, and not claims.
