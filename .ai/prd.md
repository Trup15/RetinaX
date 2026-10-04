# Product Requirements — RetinaX (improved)

## 1. Summary
RetinaX is a trustworthy, lightweight diabetic-retinopathy (DR) screening **research system**: it rejects ungradable fundus images,
grades gradable ones on the 5-point scale, explains the grade, validates explanations against expert lesion masks, calibrates
confidence, and refers uncertain cases to a specialist — evaluated across datasets, with offline/CPU feasibility measured.
It is a research prototype built on public retrospective datasets. It is **not** a medical device and makes no clinical-validation claim.

Components (unchanged from the blueprint): (1) quality gate — MobileNetV3-Small; (2) 5-stage classifier — EfficientNet-B3;
(3) uncertainty gate — temperature scaling fitted on validation data, predictive entropy, MC-Dropout; (4) XAI — Grad-CAM++, Score-CAM,
Integrated Gradients, validated with Dice/IoU/pointing-game on IDRiD lesion masks; (5) offline/edge feasibility benchmark.

## 2. Problem (motivation; keep claims cautious)
Diabetic retinopathy is a leading preventable cause of vision loss, and specialist access is limited in rural India. Published figures
(prevalence, ophthalmologist counts) must be **cited from verified sources in the paper**, not copied from this file. Limitations of existing systems
(from the blueprint §5): ungradable images are graded anyway, overconfident softmax, qualitative-only heatmaps, no abstention, single-dataset evaluation,
heavy models, fragmented pipelines. Prior rural/Indian AI-DR work exists; novelty is the integrated, rigorously evaluated framework, not "AI for rural DR".

## 3. Users
ASHA/field operator (needs: retake / pass / refer, plain language) · primary-care doctor (needs: grade, evidence, calibrated confidence) ·
ophthalmologist (needs: a referral sheet) · **researcher (primary user of v1)**: runs experiments, reads `outputs/`.

## 4. Functional requirements
- FR-1 Ingest JPG/PNG (DICOM optional later); deterministic preprocessing shared with training.
- FR-2 Quality gate: gradable vs ungradable with a validation-chosen threshold; reports sensitivity/specificity for the ungradable class.
  Blur/illumination/FOV measurements shown as *hints*; failure causes are not claimed as classes.
- FR-3 5-stage grading (0–4) with calibrated probabilities.
- FR-4 Explanations on request: Grad-CAM++ (default), Score-CAM, Integrated Gradients.
- FR-5 Lesion overlap (Dice, IoU, precision, recall, pointing game, lesion-AUROC) — **only** for images with ground-truth masks (IDRiD segmentation subset), via the research endpoint/CLI.
- FR-6 Calibrated uncertainty and referral: τ chosen on validation data for a pre-declared referral rate; output `ACCEPT_GRADE` / `SPECIALIST_REFERRAL` / `RECAPTURE_IMAGE`; printable referral sheet.
- FR-7 Cross-dataset benchmarking: APTOS internal test, IDRiD external, DDR-test external — one frozen primary model, no fine-tuning.
- FR-8 Offline operation: no network calls in the screening path; ImageNet pretrained weights are fetched only by training scripts.
- FR-9 Reproducibility: seeds, versions, configs, splits, predictions, per-image overlap tables saved for every result.

## 5. Non-functional requirements (all are measured and reported, none are promised)
| Area | Requirement | Note |
|---|---|---|
| Build | `python scripts/run_smoke_test.py` passes on a clean CPU machine, no internet | the acceptance test for the first build |
| Latency / memory / size | measured by `deploy.benchmark` and reported (median, p95, peak RSS, MB) | earlier fixed targets (≤15 MB, ≤45 ms, <1.5 s, <1 GB) are aspirations; EfficientNet-B3 fp32 is tens of MB |
| Accuracy | reported as QWK, balanced accuracy, macro-F1, per-class recall with 95 % CIs | no accuracy target is promised; internal sanity bands in `ml_protocol.md §3` are bug detectors only |
| Selective accuracy | accuracy/QWK vs referral rate table and risk–coverage curve | "> 95 % at 10–15 % referral" is hypothesis H7, to be tested |
| Safety | no DR grade is ever returned for a gate-rejected image; no grade at all when the model is not loaded | enforced by tests |

## 6. Out of scope for v1
Authentication/multi-user, cloud deployment, DICOM, EyePACS, on-device mobile app, regulatory work, fine-tuning experiments (separate, explicitly labelled).
