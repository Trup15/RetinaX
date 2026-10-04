# AI Engineer & Developer Rulebook — RetinaX

## 1. Core Engineering Principles
1. **Never Crash on Missing Weights (Cold-Start Protocol):**
   - If trained PyTorch weights (`.pth`) are not yet present in `models_weights/`, the backend **must not crash or exit**.
   - It must automatically initialize the PyTorch models with standard ImageNet weights or execute the algorithmic fallback mode (OpenCV Laplacian blur detection + green-channel vessel contrast) and log: `[WARN] Operating in baseline computer-vision mode until fine-tuned weights are present.`
2. **Never Assume Cloud Connectivity:**
   - All backend code must run completely offline without calling external inference APIs (no OpenAI, no Anthropic, no cloud endpoints in the core screening path).
3. **No Uncalibrated Softmax Probabilities:**
   - Raw neural network softmax values must never be reported to doctors as "confidence". Always run Temperature Scaling ($T = 1.38$) and report Predictive Entropy $H(p)$.
4. **Data Isolation (Zero Leakage):**
   - Training: Exclusively on **APTOS 2019** (5-fold stratified cross-validation).
   - Validation Tuning: Temperature scaling $T$ is tuned on the APTOS validation fold.
   - **IDRiD** and **DDR** are external test sets and must NEVER be used for training, learning rate adjustment, or temperature fitting.
5. **Ordinal Severity Metrics:**
   - DR severity is ordinal (0 to 4). Always report Quadratic Weighted Kappa (QWK) alongside Macro F1 and Accuracy.

---

## 2. Python & Backend Standards
- **Python Version:** 3.10 or 3.11 with strict type annotations (`typing.List`, `typing.Optional`, `Annotated`).
- **Framework:** FastAPI with `async def` endpoints. Use Pydantic v2 for data validation (`from pydantic import BaseModel, Field`).
- **Computer Vision & ML Libraries:**
  - `torch >= 2.0.0`, `torchvision >= 0.15.0`
  - `timm` for EfficientNet-B3 backbone initialization
  - `opencv-python-headless` (never install GUI `opencv-python` on headless servers)
  - `albumentations` for image augmentation and preprocessing
  - `scipy` and `scikit-learn` for ECE, QWK, and statistical tests
- **PyTorch GPU/CPU Dynamic Fallback:**
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
  Every model and tensor operation must be hardware-agnostic.
- **Image Upload Handling:** Read image bytes using `io.BytesIO` and decode using `cv2.imdecode` or `PIL.Image.open`. Do NOT write temporary files to disk unless caching.
- **Error Handling:** Use standard `HTTPException` with structured details:
  ```python
  raise HTTPException(status_code=400, detail={"code": "INVALID_IMAGE", "message": "Failed to decode fundus image."})
  ```

---

## 3. Frontend & TypeScript Standards
- **No Mock Fallback Regressions:** If the Python backend server is offline or unreachable, the frontend must smoothly fall back to the built-in client-side canvas image analyzer (`src/utils/imageAnalyzer.ts`) with a clear indicator: *"Operating in Client-Side Edge Mode"*.
- **Strict Single-Line Top Bar Contract:** Exactly 3 zones (Wordmark `RetinaX` | 3 nav links | 1 primary action button).
- **Design Constitution:**
  - Maintain the clean, clinical light theme (`bg-slate-50`, `bg-white`, `border-slate-200`).
  - No "pill sandwiches" or unneeded badges. Use clean unboxed metadata with `·` dividers.
  - Tabular numerals (`font-mono tabular-nums`) for medical probabilities and coordinates.
- **No Alert Dialogs:** Never use `window.alert()` or `window.confirm()`. Use the custom `ReferralModal.tsx` or clean toasts.

---

## 4. What the AI Should Avoid
- ❌ Do NOT add chatbots, chat bubbles, or conversational "Ask AI" widgets.
- ❌ Do NOT modify weights or thresholds using the external test sets (IDRiD / DDR).
- ❌ Do NOT store patient healthcare data without local SQLite / PostgreSQL schema validation.
- ❌ Do NOT use deprecated PyTorch methods (e.g. use `torch.no_grad()` or `torch.inference_mode()`).
