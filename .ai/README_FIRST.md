# READ FIRST — RetinaX / Trustworthy DR Screening (improved spec set)

This folder is the **improved** version of the original `.ai/` bundle. The scientific design is unchanged
(see `docs/diabetic_retinopathy_project_blueprint.md`, copied byte-for-byte). What changed is how precisely the
build is specified, so a CLI coding agent can build it, run it, and get a sensible model without guessing.
Your original files were not modified; this is a separate set.

## Order of authority (when two files disagree)
1. `docs/diabetic_retinopathy_project_blueprint.md` — the research design (what is being studied, split rules, claims).
2. `.ai/rules.md` — non-negotiable engineering/science rules.
3. `.ai/ml_protocol.md` — exact training / calibration / XAI / uncertainty recipes (the "accuracy" file).
4. `.ai/architecture.md`, `.ai/backend_integration.md`, `.ai/data_spec.md` — structure, API, data.
5. `.ai/tasks.md` — the order to build in, with a pass/fail check per task.
6. `.ai/prd.md`, `.ai/design.md`, `.ai/current_state.md`, `.ai/memory.md` — product/UI context and decisions.

## Definition of "done" for the FIRST build
The first build is **done** when `python scripts/run_smoke_test.py` exits 0 on a CPU-only machine with no dataset
downloads and no internet. That proves: preprocessing, splits, training loop, quality model, evaluation,
calibration, uncertainty, XAI + lesion validation, benchmark, and the API all run end to end.
Real-data training is a separate, later step (needs a GPU and the three datasets).

## What the agent must do FIRST (in this order)
1. Read this file, `rules.md`, `tasks.md`, `ml_protocol.md`.
2. Copy `reference/*.py` into the package (preprocessing, augmentation, metrics, config loader, synthetic data; 9 tests in `reference/test_reference.py` pass on numpy/OpenCV/sklearn) and keep their tests. Copy `templates/*` (requirements, pyproject, .gitignore, configs, smoke script) to the repo root.
3. Execute `tasks.md` top to bottom. Do not start a phase until the previous phase's check passes.
4. If something here is wrong for the installed library versions, fix the code, keep the *behaviour* described,
   and record the deviation in `.ai/memory.md` under "Deviations".

## Actions only the HUMAN can do (the agent must stop and ask, never fabricate data)
- APTOS 2019: needs a Kaggle account, Kaggle API token (`kaggle.json`) and accepting the competition rules.
- IDRiD: IEEE Dataport login (or a Kaggle mirror) — license/terms acceptance.
- DDR: **Already available locally** — 12,524 images (9,378 train + 3,146 test) in `smoke_data/DR_grading/` with `DR_grading.csv` labels. No download needed.
- A GPU (Colab / Kaggle notebook / lab machine) to train EfficientNet-B3 at 384–512 px. CPU is fine only for the smoke test.
- If any dataset is missing, the agent runs everything it can on the smoke data and clearly reports what is blocked.

## Facts to VERIFY before trusting (could not be confirmed offline; the agent must inspect the real files)
- IDRiD: the **grading** set is 516 images, but pixel-level lesion masks exist only for the **segmentation** subset
  (reported as 81 images, i.e. 54 train + 27 test). Lesion validation therefore uses ~81 images, not 516. Not every
  image has every lesion type (soft exudates in particular). Match masks to images by file name.
- APTOS: `test.csv` has **no labels**. All labelled data is the 3,662 images of `train.csv`; the internal test set
  must be carved out of it.
- DDR: **12,524 images total (9,378 train + 3,146 test), grades 0–4 only (no class 5/ungradable)**. Label file `DR_grading.csv` has `id_code`, `diagnosis`. Folder structure: `DR_grading/train/`, `DR_grading/test/`. No official train/valid/test split lists — using folder-based split.
- Kaggle mirror folder layouts differ from the official ones. Never hard-code a path before listing the folder.

## Hard "no" list (copied here because it is the most common way these projects go wrong)
- No hard-coded results anywhere in code, API, UI or docs (no fixed T, Dice, ECE, accuracy, latency).
- No DR stage ever returned by a heuristic. Without a model the API says so; it does not guess.
- No lesion-overlap numbers for an image that has no ground-truth mask.
- No tuning of anything (hyper-parameters, T, thresholds, XAI thresholds) on IDRiD or DDR-test.
