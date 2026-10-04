# Backend Integration Guide — RetinaX (improved)

Replaces the old single-file `main.py` that (a) could not be started with the documented command, (b) staged DR from pixel counts
(every synthetic image became Stage 4), (c) returned invented Dice/IoU/MC-Dropout values, (d) turned `HTTPException(400)` into a 500,
and (e) never loaded the PyTorch weights. Authoring note: the numpy/OpenCV/sklearn pieces referenced here are in `reference/` and pass
`reference/test_reference.py`. The torch/timm/FastAPI snippets below could **not** be executed in the authoring sandbox (no torch);
they are written to be correct, and the smoke tests in `tasks.md` are what proves them. Fix on first failure, keep the behaviour.

## 1. Dependencies (`templates/requirements.txt`)
Python 3.10–3.12. Install order matters on GPU machines: install the right `torch` build first (https://pytorch.org/get-started/locally),
then `pip install -r requirements.txt`.
```txt
fastapi>=0.110
uvicorn[standard]>=0.29
python-multipart>=0.0.9
pydantic>=2.6,<3
sqlalchemy>=2.0
httpx>=0.27              # FastAPI TestClient
pytest>=8
torch>=2.2
torchvision>=0.17
timm>=1.0,<2             # 1.x API: forward_features / forward_head(pre_logits=True)
grad-cam>=1.5            # pytorch-grad-cam: Grad-CAM++, Score-CAM
captum>=0.7              # Integrated Gradients
opencv-python-headless>=4.9
numpy>=1.26              # no upper cap: torch>=2.3 supports numpy 2; the old '<2.0' pin only creates wheel conflicts
scipy>=1.11
scikit-learn>=1.3        # StratifiedGroupKFold needs >=1.0
pandas>=2.0
pillow>=10
pyyaml>=6
psutil>=5.9
matplotlib>=3.8
onnx>=1.15               # optional [deploy]
onnxruntime>=1.17        # optional [deploy]
```
Removed vs old list: `aiofiles` (unused), `albumentations` (replaced by `reference/augment.py`).

## 2. API contract (v1) — all JSON; compute endpoints are plain `def`
| Method + path | Purpose |
|---|---|
| `GET /api/v1/health` | `{status, device, models: {quality: bool, dr: bool}, provenance, calibration_loaded: bool, version}` — never fails when weights are missing |
| `POST /api/v1/screen` (multipart: `file`, `eye` OD/OS default OD, `xai_method` optional: `NONE` (default) / `GRAD_CAM_PP` / `SCORE_CAM` / `INTEGRATED_GRADIENTS`, `mc_dropout` bool default false) | full pipeline |
| `POST /api/v1/quality` | quality gate only |
| `POST /api/v1/referrals` / `GET /api/v1/referrals` | store / list referral records (SQLite) |
| `GET /api/v1/results` | serves `outputs/tables/summary.json` or `{"available": false}` |
| `GET /api/v1/cohorts/idrid` and `GET /api/v1/cohorts/idrid/{name}/lesion-validation?method=GRAD_CAM_PP` | IDRiD sample images **with masks** → heatmap + real Dice/IoU/pointing-game. Only place lesion overlap exists. 404 if no masks. |
Default `xai_method` is `NONE` because Score-CAM/IG are slow; heatmaps are opt-in.

### `/screen` response (null where a step did not run)
```json
{
  "screening_id": "uuid4",
  "inference_mode": "model | quality_only",
  "model_provenance": "aptos_dev_fold0 | synthetic_smoke | null",
  "quality": {"status": "GRADABLE | UNGRADABLE | UNAVAILABLE", "p_ungradable": 0.0, "threshold": 0.0,
              "hints": {"laplacian_variance": 0.0, "illumination_uniformity": 0.0, "fov_coverage": 0.0},
              "recommended_action": "NONE | RECAPTURE_IMAGE"},
  "classification": {"predicted_stage": 0, "stage_name": "No DR", "raw_probabilities": [], "calibrated_probabilities": []},
  "uncertainty": {"method": "entropy", "temperature": 0.0, "entropy_norm": 0.0, "tau": 0.0, "is_uncertain": false,
                  "mc_dropout": {"n_passes": 30, "entropy_norm": 0.0, "mutual_information": 0.0}},
  "clinical_action": "ACCEPT_GRADE | SPECIALIST_REFERRAL | RECAPTURE_IMAGE | MODEL_NOT_LOADED",
  "reason": "null | HIGH_UNCERTAINTY | UNGRADABLE | NO_CALIBRATION | MODEL_NOT_LOADED",
  "explainability": {"method": "GRAD_CAM_PP", "target_class": 0, "heatmap_png_base64": "..."}
}
```
Rules: `temperature` and `tau` come from `artifacts/calibration.json` (if `calibration.json` is absent but a DR model is loaded → return the suggested grade with `uncertainty: null`,
`clinical_action: "SPECIALIST_REFERRAL"`, `reason: "NO_CALIBRATION"` — safety-first, and never a default number). Raw probabilities are returned for transparency but the UI shows only
calibrated ones, labelled "calibrated". No `lesion_overlap` here. "Quality hints" are measurements, not causes.

## 3. Startup / model loading (`backend/app/main.py`)
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from retinax.pipeline import RetinaXPipeline
from retinax.config import load_config
import logging
log = logging.getLogger("retinax.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()                       # env RETINAX_CONFIG / RETINAX_ARTIFACTS_DIR override defaults
    app.state.pipe = RetinaXPipeline.load(cfg)   # never raises on missing weights; flags availability instead
    if not app.state.pipe.dr_loaded:
        log.warning("[WARN] DR model not loaded; grading disabled.")
    yield

app = FastAPI(title="RetinaX API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"])
# include routers under /api/v1 ...
```
Run (repo root): `python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000`. Needs `backend/__init__.py` and `backend/app/__init__.py`.

## 4. `RetinaXPipeline` essentials (in `src/retinax/pipeline.py`)
```python
class RetinaXPipeline:
    @classmethod
    def load(cls, cfg): ...
        # for each of quality / dr: if file exists -> torch.load(..., map_location="cpu", weights_only=True),
        # rebuild with timm.create_model(meta["arch"], pretrained=False, num_classes=meta["num_classes"], drop_rate=meta.get("drop_rate", 0.0)),
        # model.load_state_dict(ck["state_dict"]); model.eval(); .to(device)
        # calibration.json / quality_threshold.json -> dicts or None
    def screen(self, img_bgr, xai_method=None, mc_dropout=False) -> dict: ...   # logic in architecture.md §4
```
Quality index mapping (a past bug source): quality model output index **1 = ungradable**, so `p_ungradable = softmax(logits)[1]`; `status = UNGRADABLE if p_ungradable >= threshold`.
The DR model returns **logits**; temperature scaling is `softmax(logits / T)` using `softmax` from `reference/uncertainty_metrics.py`.

MC-Dropout (head-only, backbone run once):
```python
@torch.inference_mode()
def mc_dropout_probs(model, x, n=30, T=1.0):
    model.eval()                                   # BatchNorm stays in eval mode
    feats = model.forward_head(model.forward_features(x), pre_logits=True)   # (B, C)
    p = float(getattr(model, "drop_rate", 0.3))    # same rate used in training
    clf = model.get_classifier()
    outs = [torch.softmax(clf(torch.nn.functional.dropout(feats, p=p, training=True)) / T, dim=-1)
            for _ in range(n)]
    return torch.stack(outs)                       # (n, B, K) -> mean prob, entropy, mutual information
```
If your timm version's `forward_head(pre_logits=True)` already applies dropout, call `model.eval()` first (it does nothing in eval) - the smoke test asserts
`mc.std(0).max() > 0` (passes differ) so a wrong assumption is caught immediately.

Quality "hints" (measurements): Laplacian variance on the central region of the model-space image (`reference` has the formulas in the old doc; keep them, but
they are **not** used to decide gradability when the quality model is loaded). If the quality model is missing, the gate falls back to
a Laplacian-variance rule whose cut-off is read from `quality_threshold.json["laplacian_fallback"]` (fitted on DDR valid) — absent → status `UNAVAILABLE`.

## 5. Error handling pattern
```python
try:
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, {"code": "INVALID_IMAGE", "message": "Failed to decode fundus image."})
    ...
except HTTPException:
    raise                                   # do NOT swallow into a 500
except Exception as e:
    log.exception("inference failed")
    raise HTTPException(500, {"code": "INFERENCE_ERROR", "message": str(e)})
```
Also: reject uploads > 15 MB (413), content-type not image/* (415), images < 128 px on a side (400).

## 6. Tests the backend must ship (`tests/`)
`test_api.py` with FastAPI `TestClient`: (1) `/health` 200 with **no** artifacts present, (2) `/screen` with no artifacts → `clinical_action == "MODEL_NOT_LOADED"` and no `classification`,
(3) invalid bytes → 400 (not 500), (4) with smoke artifacts: `/screen` returns all keys, probabilities sum to 1 ± 1e-6, `provenance == "synthetic_smoke"`,
(5) black image / tiny image handled without exception, (6) `/results` returns `{"available": false}` when no summary exists.
