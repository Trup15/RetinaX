# Current State & Handoff — RetinaX (improved)

## What exists (as of this spec)
- Research design: complete (`docs/diabetic_retinopathy_project_blueprint.md`). Status per the blueprint: **proposed methodology; no trained models, no measured results.**
- This spec set, plus tested reference modules in `reference/` (preprocessing, augmentation, calibration/uncertainty/referral metrics, lesion metrics, synthetic data) with 8 passing tests (numpy/OpenCV/sklearn only).
- The earlier bundle described a React frontend as "implemented", but **no frontend source was included** in the handoff, so the agent cannot assume it exists. Check for `frontend/`; if missing, create it (Phase 9).

## What does NOT exist yet (the agent builds it)
Everything under `src/retinax/`, `backend/`, `tests/`, `scripts/run_smoke_test.py`, trained weights, calibration/threshold files, any experiment outputs.

## What is NOT a result (must not appear anywhere as a finding)
Any temperature value, ECE, Dice/IoU, pointing-game rate, accuracy, selective accuracy, latency or model-size figure that appeared in earlier drafts
(e.g. "T = 1.38", "Dice 0.534", "97.8 %", "42 ms", "15 MB"). They were placeholders; they are removed or marked as hypotheses.

## Immediate next step
Run Phase 0 → Phase 7 of `tasks.md`. The first deliverable is a green smoke test, not a trained model.
