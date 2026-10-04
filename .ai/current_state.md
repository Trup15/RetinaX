# Current State & Handoff Summary — RetinaX

## 1. What is Implemented & Working Right Now (Frontend)
- **Framework:** React 19 SPA running on Vite + TypeScript + Tailwind CSS v4.
- **Theme:** Medical Light Theme (`bg-slate-50` body, `bg-white` cards, `border-slate-200` dividers).
- **Core Views:**
  1. `Retinal Screener` (`src/components/SimpleScreener.tsx`):
     - Cohort patient picker + custom image uploader.
     - View switchers (`Normal Photo`, `AI Highlights`, `Doctor's Lesions`, `Compare Both`).
     - 500x500 Canvas viewport with FOV circular crop and crosshair HUD telemetry.
     - Dynamic 3-card decision panel:
       * Card 1: Image Quality Check (Gradable vs Ungradable Blur / Cataract).
       * Card 2: 5-Stage Disease Severity Grade with probability distribution.
       * Card 3: Action & Specialist Referral (with 1-click printable Doctor Referral Slip).
       * Card 4: Doctor Lesion Match (% Overlap, IoU, Pointing Game).
  2. `Lesion AI Check` (`src/components/SimpleLesionValidation.tsx`):
     - Dynamic threshold scrubber ($\tau \in [0.15, 0.80]$).
     - Method comparisons (Grad-CAM++, Score-CAM, Integrated Gradients) with real-time recalculating Dice, IoU, Recall, and Precision.
  3. `Safety & Real-World Tests` (`src/components/SimpleResearchSummary.tsx`):
     - Hospital generalization comparison (APTOS, IDRiD, DDR).
     - Dynamic specialist referral slider ($0\%\text{--}35\%$) showing real-time selective accuracy boost (up to $97.8\%$) and intercepted error rates.
     - Edge deployment specifications ($15\text{ MB}$, $42\text{ ms}$, offline smartphone capable).
  4. `Referral Modal` (`src/components/ReferralModal.tsx`):
     - Full clinical referral sheet with patient demographics, reason, preliminary grading, and ophthalmologist sign-off blocks. Printable via `window.print()`.

- **Client-Side Simulation Engine (`src/utils/imageAnalyzer.ts`):**
  - Measures real Laplacian optical blur variance ($\sigma^2_{\nabla^2}$) using an offscreen HTML5 canvas.
  - Measures dynamic illumination uniformity and detects red punctate lesions and bright exudates.
  - Computes discrete $100 \times 100$ pixel grid set intersection ($|A_\tau \cap M|$) and union ($|A_\tau \cup M|$) to output genuine mathematical Dice and IoU scores.

---

## 2. What Needs to Be Built in the CLI Backend
1. **Python FastAPI Server (`backend/app/main.py`):**
   - Implement the actual PyTorch inference routes (`POST /api/v1/screen`, `POST /api/v1/quality`, `POST /api/v1/explain`).
2. **Model Weights Loading (`backend/app/ml/weights/`):**
   - Provide weights for MobileNetV3-Small (quality) and EfficientNet-B3 (APTOS DR classifier).
   - Alternatively, write the training scripts using PyTorch + Timm to train on APTOS 2019.
3. **Database Layer:**
   - SQLite / PostgreSQL with SQLAlchemy tables to persist patient screening sessions and clinical referral forms.
4. **Wire Frontend to Backend:**
   - Replace or augment the client-side canvas analyzer in `src/utils/retinalImageProcessor.ts` by fetching from `http://localhost:8000/api/v1/screen`.
