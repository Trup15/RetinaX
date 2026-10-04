# Implementation Roadmap & Verifiable Tasks — RetinaX

Each task has an **Execution Command** and a **Verification Check** so the CLI AI agent can confirm success before moving to the next task.

---

## Phase 1: Environment & Server Scaffolding
- [ ] **Task 1.1: Virtualenv & Dependency Installation**
  - *Action:* Create `backend/requirements.txt` from `.ai/backend_integration.md` and install packages.
  - *Command:* `python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
  - *Verification:* `python -c "import fastapi, torch, timm, cv2; print('Dependencies OK')"`
- [ ] **Task 1.2: FastAPI Entrypoint & Health Route**
  - *Action:* Create `backend/app/main.py` with `/api/v1/health`.
  - *Command:* `python -m uvicorn app.main:app --port 8000 &`
  - *Verification:* `curl -s http://localhost:8000/api/v1/health | grep '"status":"healthy"'`

---

## Phase 2: Core Vision & Quality Gating
- [ ] **Task 2.1: Circular FOV Cropping & Preprocessing**
  - *Action:* Implement `crop_circular_fov(img)` using Otsu thresholding in OpenCV.
  - *Verification:* Pass a test black-bordered fundus image and assert output shape is `(512, 512, 3)` with $>90\%$ of black margin removed.
- [ ] **Task 2.2: Optical Blur (Laplacian Variance) Check**
  - *Action:* Implement `compute_optical_sharpness(img)`.
  - *Verification:* Assert that Gaussian-blurred test image yields variance $< 15.0$ and triggers `UNGRADABLE` status.

---

## Phase 3: 5-Stage Classification & Temperature Calibration
- [ ] **Task 3.1: Logits & Softmax Formulation**
  - *Action:* Implement 5-class forward pass with classes 0 (No DR) to 4 (PDR).
  - *Verification:* Ensure sum of probabilities $\sum p_i = 1.0 \pm 10^{-6}$.
- [ ] **Task 3.2: Temperature Scaler ($T = 1.38$) & Shannon Entropy**
  - *Action:* Implement scalar division $z_i / T$ and $H(p) = -\sum p_i \log_2(p_i)$.
  - *Verification:* Test boundary logits `[1.0, 3.2, 3.1, 0.5, -1.0]` and confirm $H(p) > 0.85\text{ b}$, correctly triggering `SPECIALIST_REFERRAL`.

---

## Phase 4: Full Pipeline REST Endpoint
- [ ] **Task 4.1: Endpoint `POST /api/v1/screen`**
  - *Action:* Accept multipart form data with image file, eye (`OD`/`OS`), and `xai_method`.
  - *Verification Command:*
    ```bash
    curl -X POST "http://localhost:8000/api/v1/screen" \
      -F "file=@sample_eye.jpg" \
      -F "eye=OD" \
      -F "xai_method=GRAD_CAM_PP"
    ```
  - *Expected JSON keys:* `quality`, `classification`, `uncertainty`, `explainability`.

---

## Phase 5: Quantitative Lesion Overlap (IDRiD Evaluation)
- [ ] **Task 5.1: Discrete Grid Dice & IoU Calculator**
  - *Action:* Implement pixel set intersection and union on $100 \times 100$ activation matrices.
  - *Verification:* Check mathematical identity: $\text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}$.

---

## Phase 6: Frontend Bridge & End-to-End Verification
- [ ] **Task 6.1: Connect `src/utils/apiClient.ts`**
  - *Action:* Ensure frontend calls `http://localhost:8000/api/v1/screen` when user uploads or selects an image.
  - *Verification:* In browser DevTools Network tab, observe successful `200 OK` response from `localhost:8000`.
- [ ] **Task 6.2: Offline / Fallback Integrity Test**
  - *Action:* Terminate the Python server (`kill %1`) and refresh the frontend.
  - *Verification:* Confirm the frontend seamlessly continues working using the internal `src/utils/imageAnalyzer.ts` without white-screening or crashing.
