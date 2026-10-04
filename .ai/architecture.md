# System Architecture — RetinaX

## 1. High-Level Architectural Diagram

```
+-----------------------------------------------------------------------------------+
|                                 CLIENT TIER                                       |
|  React 19 + TypeScript + Tailwind CSS (Vite SPA)                                  |
|  - Viewport Canvas (500x500 FOV, CLAHE, Jet Colormap XAI Overlay)                 |
|  - Patient Screening Console (Sample Cohorts & Drag-and-Drop Image Uploader)       |
|  - Specialist Referral Generator (Printable Clinical Sheet)                       |
|  - Interactive Benchmark Labs (Dice/IoU Scrubber, Risk-Coverage Simulator)         |
+------------------------------------------+----------------------------------------+
                                           | HTTP REST API (JSON + Multipart)
                                           v
+-----------------------------------------------------------------------------------+
|                                BACKEND TIER                                       |
|  FastAPI (Python 3.10+) + Uvicorn Async Server                                    |
|                                                                                   |
|  [ Router Tier: /api/v1 ]                                                         |
|  +-- /screen          : Upload fundus image -> Full inference pipeline           |
|  +-- /quality         : MobileNetV3-Small quality gate check only                 |
|  +-- /classify        : EfficientNet-B3 5-stage inference + Temperature Scaling   |
|  +-- /explain         : Grad-CAM++ / Score-CAM / Integrated Gradients heatmap    |
|  +-- /validate-lesion : Calculate Dice / IoU against IDRiD lesion masks          |
|  +-- /referrals       : Store and list specialist referral records               |
|                                                                                   |
|  [ Machine Learning & Computer Vision Services ]                                 |
|  +-- QualityGateService     : MobileNetV3-Small (Weights: quality_mobilenetv3.pt) |
|  +-- DRClassifierService    : EfficientNet-B3 (Weights: dr_efficientnet_b3.pt)     |
|  +-- CalibrationEngine      : Temperature Scaling (T = 1.38) + Shannon Entropy   |
|  +-- SaliencyEngine         : PyTorch hooks for Grad-CAM++ & Integrated Gradients |
|  +-- LesionOverlapService   : Discrete grid set-intersection & Pointing Game      |
|  +-- PreprocessingService   : Circular FOV mask, auto-crop, CLAHE (OpenCV)        |
+------------------------------------------+----------------------------------------+
                                           | SQLAlchemy ORM
                                           v
+-----------------------------------------------------------------------------------+
|                                 DATA PERSISTENCE                                  |
|  SQLite (Local Dev / Edge Kiosk) OR PostgreSQL (Hospital Deployment)              |
|  - patients: id, age, gender, medical_record_num                                  |
|  - screenings: id, patient_id, eye, image_path, quality_status, stage, entropy,   |
|               temperature, referral_status, xai_heatmap_path, created_at          |
|  - lesion_annotations: id, screening_id, lesion_type, polygon_geojson             |
|  - model_checkpoints: id, model_name, version, fp32_size_mb, int8_size_mb, ece    |
+-----------------------------------------------------------------------------------+
```

---

## 2. Recommended Repository Structure (Monorepo)

```
retinax-root/
+-- .ai/                          # Agent specifications & memory
|   +-- prd.md
|   +-- architecture.md
|   +-- rules.md
|   +-- design.md
|   +-- tasks.md
|   +-- memory.md
|   +-- current_state.md
|   +-- data_spec.md
|   +-- backend_integration.md
+-- frontend/                     # React 19 + TypeScript + Vite (Current App)
|   +-- src/
|   |   +-- components/
|   |   |   +-- SimpleScreener.tsx
|   |   |   +-- RetinalCanvasViewer.tsx
|   |   |   +-- SimpleLesionValidation.tsx
|   |   |   +-- SimpleResearchSummary.tsx
|   |   |   +-- ReferralModal.tsx
|   |   +-- types/
|   |   |   +-- pipeline.ts
|   |   +-- utils/
|   |   |   +-- retinalImageProcessor.ts
|   |   |   +-- imageAnalyzer.ts
|   |   |   +-- apiClient.ts      # Connects frontend to backend API
|   |   +-- data/
|   |   |   +-- sampleCohorts.ts
|   |   |   +-- researchBenchmarks.ts
|   |   +-- App.tsx
|   |   +-- main.tsx
|   +-- package.json
|   +-- vite.config.ts
+-- backend/                      # Python FastAPI + PyTorch Backend
|   +-- app/
|   |   +-- api/
|   |   |   +-- v1/
|   |   |       +-- endpoints/
|   |   |           +-- screening.py
|   |   |           +-- quality.py
|   |   |           +-- explainability.py
|   |   |           +-- referrals.py
|   |   |           +-- cohorts.py
|   |   |       +-- api_router.py
|   |   +-- core/
|   |   |   +-- config.py         # App settings & env vars
|   |   |   +-- security.py
|   |   +-- db/
|   |   |   +-- session.py        # SQLAlchemy database engine
|   |   |   +-- models.py         # Patient, Screening, Referral tables
|   |   +-- ml/
|   |   |   +-- models/
|   |   |   |   +-- mobilenet_quality.py   # MobileNetV3-Small architecture
|   |   |   |   +-- efficientnet_dr.py     # EfficientNet-B3 architecture
|   |   |   +-- weights/
|   |   |   |   +-- quality_gate.pth       # Trained on DDR / EyePACS ungradables
|   |   |   |   +-- efficientnet_b3_aptos.pth # Trained on APTOS 2019
|   |   |   +-- xai/
|   |   |   |   +-- gradcam_pp.py          # Grad-CAM++ implementation
|   |   |   |   +-- scorecam.py            # Score-CAM implementation
|   |   |   |   +-- integrated_gradients.py # Integrated Gradients
|   |   |   +-- calibration/
|   |   |   |   +-- temperature_scaling.py # Temperature scaler (T=1.38)
|   |   |   |   +-- mc_dropout.py          # Monte Carlo Dropout sampler
|   |   |   +-- evaluation/
|   |   |       +-- lesion_metrics.py      # Dice, IoU, Pointing Game on IDRiD
|   |   |   +-- preprocessing/
|   |   |       +-- fov_crop.py            # Circular FOV mask & black margin crop
|   |   |       +-- clahe.py               # Green channel CLAHE
|   |   +-- schemas/
|   |   |   +-- screening.py      # Pydantic request/response models
|   |   |   +-- referral.py
|   |   +-- services/
|   |   |   +-- pipeline_service.py # Orchestrates Quality -> DR -> XAI -> Referral
|   |   +-- main.py               # FastAPI application entrypoint
|   +-- tests/
|   |   +-- test_quality.py
|   |   +-- test_classifier.py
|   |   +-- test_xai.py
|   +-- Dockerfile
|   +-- requirements.txt
+-- models_weights/               # Directory for trained .pt / .onnx / .tflite weights
+-- datasets/                     # Directory for dataset paths (APTOS, IDRiD, DDR)
```

---

## 3. Core API Endpoints

### 3.1 `POST /api/v1/screen`
- **Description:** Runs the complete end-to-end pipeline on an uploaded fundus image.
- **Request:** `multipart/form-data` with `file: UploadFile`, `eye: 'OD' | 'OS'`, `patient_id: Optional[str]`, `xai_method: 'GRAD_CAM_PP' | 'SCORE_CAM' | 'INTEGRATED_GRADIENTS'`.
- **Response:**
```json
{
  "screening_id": "scr_9823482",
  "quality": {
    "status": "GRADABLE",
    "gradable_probability": 0.974,
    "ungradable_probability": 0.026,
    "optical_sharpness_variance": 42.1,
    "illumination_uniformity": 0.94
  },
  "classification": {
    "predicted_stage": 2,
    "stage_name": "Moderate Non-Proliferative DR",
    "calibrated_confidence": 0.784,
    "raw_probabilities": [0.03, 0.12, 0.78, 0.05, 0.02],
    "calibrated_probabilities": [0.05, 0.14, 0.72, 0.06, 0.03]
  },
  "uncertainty": {
    "temperature": 1.38,
    "predictive_entropy": 0.642,
    "entropy_threshold": 0.85,
    "is_uncertain": false,
    "mc_dropout_variance": 0.0034,
    "clinical_action": "ACCEPT_DIAGNOSIS"
  },
  "explainability": {
    "method": "GRAD_CAM_PP",
    "heatmap_data_url": "data:image/png;base64,iVBORw0KGgo...",
    "heatmap_base64": "data:image/png;base64,iVBORw0KGgo...",
    "lesion_overlap": {
      "dice": 0.542,
      "iou": 0.371,
      "pointing_game_hit": true,
      "recall": 0.694
    }
  }
}
```

### 3.2 `POST /api/v1/referrals`
- **Description:** Stores and retrieves an official clinical referral sheet for uncertain or ungradable patients.
