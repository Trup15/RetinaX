# Product Requirements Document (PRD) — RetinaX

## 1. Executive Summary
**RetinaX** (formerly SafeDR / RetinaTrust AI) is a clinically grounded, trustworthy, and lightweight deep-learning system designed for Diabetic Retinopathy (DR) screening in resource-constrained environments (specifically rural Indian primary health centers and mobile eye clinics).

Unlike traditional black-box classifiers that force an autonomous prediction on every eye, RetinaX implements a **Dual-Safety-Gated Architecture**:
1. **Safety Gate 1 (Optical Quality Gate - MobileNetV3-Small):** Automatically detects and rejects ungradable images (optical blur, cataract media opacity, illumination non-uniformity), prompting recapture rather than outputting false negative misdiagnoses.
2. **5-Stage Severity Classifier (EfficientNet-B3):** Categorizes gradable fundus photographs according to the International Clinical Diabetic Retinopathy (ICDR) scale (Stages 0–4).
3. **Safety Gate 2 (Uncertainty-Aware Specialist Referral):** Uses validation-fitted Temperature Scaling ($T = 1.38$), predictive Shannon entropy ($H(p)$), and Monte Carlo Dropout to identify boundary or ambiguous cases, routing them to ophthalmologists with a printable specialist referral slip.
4. **Lesion-Grounded Explainable AI (XAI):** Generates saliency heatmaps (Grad-CAM++, Score-CAM, Integrated Gradients) and quantitatively benchmarks them against expert ophthalmologist pixel-level lesion annotations on the Indian IDRiD benchmark using Dice, IoU, and Pointing-Game localization.
5. **Edge Feasibility:** The end-to-end quantized pipeline requires $\le 15.0\text{ MB}$ INT8 memory and runs in $\le 45\text{ ms}$ on standard smartphone CPUs without internet.

---

## 2. Problem Statement
- **Epidemic Scale:** Over 77 million people in India live with diabetes; diabetic retinopathy is a leading cause of preventable blindness.
- **Specialist Scarcity:** There are fewer than 25,000 ophthalmologists in India, with $>70\%$ practicing in urban centers, leaving rural populations unreached.
- **Failures of Existing AI Systems:**
  - *Ungradable Blindspot:* They force a prediction on blurry or cataractous images, frequently diagnosing severe disease as "Normal".
  - *Overconfidence:* Softmax probabilities are notoriously uncalibrated, outputting 99% confidence on out-of-distribution or erroneous samples.
  - *Unvalidated Saliency Slop:* Visual explanations are displayed as subjective color blobs without measuring whether they align with actual microaneurysms or hemorrhages.
  - *Cloud Dependency:* Models requiring high-end GPUs cannot run in rural villages with frequent blackouts and zero cellular connectivity.

---

## 3. Target Users & Personas
1. **ASHA Workers / Rural Health Operators:** Non-expert frontline health workers operating portable smartphone or handheld non-mydriatic fundus cameras.
   - *Needs:* Immediate quality feedback ("Retake photo: too blurry"), binary screening action (Pass vs Refer), zero medical jargon.
2. **Primary Care Physicians (MBBS):** Rural clinic doctors reviewing screening batches.
   - *Needs:* 5-stage classification, visual evidence of lesions, clear confidence indicator.
3. **Consultant Ophthalmologists (Retina Specialists):** Receiving referred patients.
   - *Needs:* Printable clinical referral sheet with eye examination telemetry, suspected stage, optical defect notes, and lesion coordinates.

---

## 4. Core Functional Requirements
- **FR-1: Image Ingestion & Preprocessing:** Upload RGB retinal fundus images (JPG/PNG/DICOM), execute circular Field-of-View (FOV) cropping to eliminate black borders, and optional green-channel Contrast Limited Adaptive Histogram Equalization (CLAHE).
- **FR-2: Automated Quality Gate:** Determine Gradable vs Ungradable ($\ge 96\%$ sensitivity). Classify failure mode into *Optical Blur* vs *Media Opacity Cataract* vs *Illumination Fault*.
- **FR-3: 5-Stage Disease Classification:** Output probabilities across Stages 0 (No DR), 1 (Mild NPDR), 2 (Moderate NPDR), 3 (Severe NPDR), 4 (Proliferative DR).
- **FR-4: Visual Explanation Generation:** Real-time generation of Grad-CAM++, Score-CAM, and Integrated Gradients heatmaps.
- **FR-5: Quantitative Lesion Overlap:** Calculate real pixel-grid Dice Similarity Coefficient, Intersection over Union (IoU), and Pointing-Game accuracy against ground-truth lesion masks (Microaneurysms, Blot Hemorrhages, Hard Exudates, Cotton Wool Spots).
- **FR-6: Calibrated Uncertainty & Specialist Triage:** Calculate temperature-scaled probabilities ($T = 1.38$) and Shannon entropy $H(p)$. If $H(p) > \tau_{\text{ref}}$ (default $0.85\text{ b}$), mark as `REFERRAL_RECOMMENDED` and generate a printable PDF/Sheet.
- **FR-7: Cross-Dataset Benchmarking:** Maintain evaluation metrics across 3 benchmark cohorts: APTOS 2019 (Internal), IDRiD (Indian external), and DDR (Broad shifted external).
- **FR-8: Offline / Edge Operation:** Complete backend inference must operate locally on CPU/Edge hardware without third-party cloud API dependencies.

---

## 5. Non-Functional Requirements
- **Latency:** End-to-end inference (Quality + Classification + Heatmap) $< 1.5\text{ s}$ on quad-core CPU, $< 200\text{ ms}$ on CUDA GPU.
- **Memory Footprint:** Peak RAM consumption $< 1.0\text{ GB}$.
- **Accuracy Target:** Selective accuracy on retained cohort $> 95\%$ when referring top $10\text{--}15\%$ uncertain cases.
- **Safety Protocol:** Zero false assurance on ungradable images.
