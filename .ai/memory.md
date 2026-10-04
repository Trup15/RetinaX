# Project Memory & Architecture Decision Records (ADR) — RetinaX

This document captures historical decisions, trade-offs, and critical context so that CLI models and developers do not re-litigate established decisions.

---

## 1. Architectural Decisions Log

### ADR-01: Selection of Dual-Gated Pipeline over Monolithic Classifier
- **Context:** Standard DR architectures attempt to classify all images directly into 5 stages.
- **Decision:** Split into an explicit **Quality Gate (MobileNetV3-Small)** and **DR Classifier (EfficientNet-B3)** followed by an **Uncertainty Gate**.
- **Rationale:** Feeding ungradable blurry or cataract images into a disease classifier causes catastrophic false-negative errors (diagnosing severe retinopathy as "Normal" because vessels cannot be seen). Rejecting ungradables first protects patient safety.

### ADR-02: Backbone Model Choices (MobileNetV3-Small & EfficientNet-B3)
- **Quality Gate:** MobileNetV3-Small ($\approx 2.54\text{ M}$ parameters, $2.6\text{ MB}$ INT8). Highly sensitive to optical high-frequency blur while adding under $9\text{ ms}$ CPU overhead.
- **DR Classifier:** EfficientNet-B3 ($\approx 12.23\text{ M}$ parameters, $12.4\text{ MB}$ INT8). Optimal resolution ($300\text{--}512\text{ px}$) to resolve microaneurysms ($< 50\,\mu\text{m}$) without the computational bloat of EfficientNet-B7 or ViT.

### ADR-03: Confidence Calibration via Temperature Scaling ($T = 1.38$)
- **Context:** Modern neural networks are miscalibrated and produce unreliably high softmax probabilities.
- **Decision:** Optimize a single temperature scalar $T$ on the validation set using negative log-likelihood loss with L-BFGS.
- **Result:** $T = 1.38$ reduced Expected Calibration Error (ECE) from $8.9\%$ to $3.8\%$ on APTOS, and from $14.2\%$ to $6.4\%$ on the external IDRiD dataset. Temperature scaling preserves class ranking (argmax unchanged) while generating trustworthy predictive entropy.

### ADR-04: Saliency Method Benchmark on IDRiD
- **Context:** Grad-CAM is often criticized as visual pseudoscience in medical imaging.
- **Findings:** Testing on IDRiD expert lesion masks revealed:
  - **Grad-CAM++** scored highest ($0.534$ Dice, $0.364$ IoU, $86\%$ Pointing Game accuracy).
  - **Score-CAM** scored second ($0.472$ Dice); produces smoother heatmaps but bleeds onto healthy retina.
  - **Integrated Gradients** scored lowest ($0.395$ Dice) due to pixel-level baseline scatter.
- **Decision:** Set Grad-CAM++ as the default clinical explanation layer.

### ADR-05: Client-Side Fallback Engine (`imageAnalyzer.ts`)
- **Context:** The frontend is frequently demonstrated in web environments before the Python backend is running.
- **Decision:** Built a complete client-side canvas pixel analyzer that computes real Laplacian variance ($\sigma^2_{\nabla^2}$), illumination uniformity, and real pixel-grid Dice/IoU set operations. The frontend automatically switches to the Python backend when detected at `http://localhost:8000`.

---

## 2. Key Nomenclature & Constants
- **Project Name:** `RetinaX`
- **Temperature Constant:** $T = 1.38$
- **Entropy Referral Cutoff:** $\tau_{\text{entropy}} = 0.85\text{ bits}$
- **Laplacian Blur Threshold:** $\sigma^2 < 18.0$ (values below 18 are flagged as ungradable blur)
- **Standard Image Dimensions:** $500 \times 500$ viewport, $512 \times 512$ model input tensor
