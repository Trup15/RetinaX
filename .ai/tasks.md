# Implementation Roadmap & Task Breakdown — RetinaX

This document breaks down the end-to-end implementation for CLI-based AI coding agents (Claude Code, Cursor, Aider, Copilot, etc.) into sequential, verifiable phases.

---

## Phase 1: Environment & Backend Scaffolding
- [ ] **Task 1.1:** Initialize the `backend/` directory with `pyproject.toml` or `requirements.txt` (`fastapi`, `uvicorn`, `torch`, `torchvision`, `timm`, `opencv-python-headless`, `albumentations`, `scipy`, `scikit-learn`, `sqlalchemy`, `pydantic`).
- [ ] **Task 1.2:** Set up the FastAPI server with CORS middleware allowing `http://localhost:3000` (or Vite dev port) and healthcheck endpoint `GET /api/v1/health`.
- [ ] **Task 1.3:** Configure SQLite local database using SQLAlchemy with tables: `patients`, `screenings`, `referrals`.

---

## Phase 2: Preprocessing & FOV Cropping Service
- [ ] **Task 2.1:** Implement `backend/app/ml/preprocessing/fov_crop.py`:
  - Detect circular pupil mask using Otsu thresholding / Hough circle transform.
  - Crop black bounding borders while preserving retinal anatomy.
  - Standardize all inputs to $512 \times 512$ resolution.
- [ ] **Task 2.2:** Implement CLAHE contrast enhancement on the green channel using OpenCV (`cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`).

---

## Phase 3: Image Quality Gate Service (MobileNetV3-Small)
- [ ] **Task 3.1:** Implement model definition in `backend/app/ml/models/mobilenet_quality.py` using `timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=2)`.
- [ ] **Task 3.2:** Add inference wrapper that outputs `is_gradable: bool`, `gradable_probability: float`, and failure mode classification (`OPTICAL_BLUR`, `CATARACT_OPACITY`, `ILLUMINATION_FAULT`).
- [ ] **Task 3.3:** Add unit tests verifying that blurred fundus images are correctly rejected.

---

## Phase 4: DR 5-Stage Classifier Service (EfficientNet-B3)
- [ ] **Task 4.1:** Implement model definition in `backend/app/ml/models/efficientnet_dr.py` using `timm.create_model('efficientnet_b3', pretrained=True, num_classes=5)`.
- [ ] **Task 4.2:** Implement forward pass producing raw unnormalized logits `[z_0, z_1, z_2, z_3, z_4]` and standard softmax distribution.
- [ ] **Task 4.3:** Implement training / fine-tuning script on APTOS 2019 using Cross-Entropy Loss with Label Smoothing ($0.1$) and Stratified 5-Fold Cross-Validation.

---

## Phase 5: Confidence Calibration & Uncertainty Referral
- [ ] **Task 5.1:** Implement `backend/app/ml/calibration/temperature_scaling.py`:
  - Optimization function using L-BFGS to find optimal scalar $T$ on validation logits ($T = 1.38$).
  - Softmax calculation with scaled logits: $p_i = \frac{e^{z_i/T}}{\sum e^{z_j/T}}$.
  - Expected Calibration Error (ECE) metric calculator with 10 reliability bins.
- [ ] **Task 5.2:** Implement Shannon Predictive Entropy calculation: $H(p) = -\sum_{i=0}^4 p_i \log_2(p_i)$.
- [ ] **Task 5.3:** Implement Monte Carlo Dropout sampler ($N = 30$ forward passes with `model.train()` mode enabled for dropout layers).
- [ ] **Task 5.4:** Implement triage logic: If $H(p) > 0.85\text{ b}$, return `SPECIALIST_REFERRAL` status.

---

## Phase 6: Explainable AI & Lesion Grounding Engine
- [ ] **Task 6.1:** Implement **Grad-CAM++** in `backend/app/ml/xai/gradcam_pp.py`:
  - Hook into the final convolutional feature layer of EfficientNet-B3 (`conv_head`).
  - Calculate positive partial gradient weights and generate 2D attribution heatmap.
- [ ] **Task 6.2:** Implement **Score-CAM** and **Integrated Gradients** for comparative benchmarks.
- [ ] **Task 6.3:** Implement quantitative lesion evaluation against IDRiD pixel masks in `backend/app/ml/evaluation/lesion_metrics.py`:
  - Compute discrete set Intersection over Union (IoU) and Dice Similarity Coefficient.
  - Implement Pointing-Game localization check ($\arg\max XAI \in \text{Lesion}$).

---

## Phase 7: REST API Endpoints & Persistence
- [ ] **Task 7.1:** Implement `POST /api/v1/screen` endpoint orchestrating:
  1. FOV crop $\to$ 2. Quality Gate $\to$ 3. (If gradable) DR Classifier $\to$ 4. Temperature Scaling $\to$ 5. Grad-CAM++ generation $\to$ 6. Return unified JSON.
- [ ] **Task 7.2:** Implement `GET /api/v1/screenings/{id}` and `POST /api/v1/referrals` to persist patient referral documents.
- [ ] **Task 7.3:** Implement `GET /api/v1/benchmarks` returning cross-dataset metrics (APTOS, IDRiD, DDR).

---

## Phase 8: Frontend-Backend Integration
- [ ] **Task 8.1:** Create `frontend/src/utils/apiClient.ts` with Axios/Fetch functions to call `POST /api/v1/screen`.
- [ ] **Task 8.2:** Update `SimpleScreener.tsx` to call the live FastAPI backend when an image is selected/uploaded, with automatic fallback to client-side canvas analysis if server is unreachable.
- [ ] **Task 8.3:** Connect the "Download / Print Doctor Referral Slip" button to save referral records to the backend database.
