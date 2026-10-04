# Design Constitution & UI/UX Style Guide — RetinaX

## 1. Design Philosophy
RetinaX adheres to the **Science & Biotech High-Precision Design Constitution**:
- **Aesthetic:** Pristine, clinical, reassuring, and distraction-free.
- **Theme:** Medical Light Theme (`bg-slate-50` primary canvas, `bg-white` structural cards, `border-slate-200` hairline dividers).
- **Core Principle:** A frontline health worker or rural doctor must be able to understand the screening result in **under 5 seconds** without reading a wall of machine learning jargon.

---

## 2. Color Palette (60-30-10 Discipline)

| Role | Tailwind Class | Hex Value | Usage |
|---|---|---|---|
| **60% Primary Canvas** | `bg-slate-50` | `#F8FAFC` | Global viewport background, page body |
| **30% Structural Panels** | `bg-white` | `#FFFFFF` | Cards, modals, top navigation bar, tables |
| **Sub-panels & Controls** | `bg-slate-100` / `bg-slate-50` | `#F1F5F9` | Progress bar tracks, stat boxes, chip buttons |
| **Borders & Dividers** | `border-slate-200` | `#E2E8F0` | Card borders, section separators, table grids |
| **10% Clinical Accent (Primary)** | `text-cyan-700` / `bg-cyan-600` | `#0891B2` | Brand wordmark, active navigation tabs, action buttons |
| **Success / Normal Retinopathy** | `text-emerald-700` / `bg-emerald-600` | `#059669` | Stage 0 (No DR), Gradable pass, high confidence |
| **Caution / Moderate Retinopathy** | `text-amber-700` / `bg-amber-500` | `#D97706` | Stage 2 (Moderate DR), Specialist referral alert |
| **High Threat / Ungradable** | `text-rose-700` / `bg-rose-600` | `#E11D48` | Stage 4 (PDR), Image quality rejection, optical blur |

---

## 3. Typography Hierarchy
1. **Primary Interface Font:** `Plus Jakarta Sans`, system-ui, sans-serif
   - Headings: `font-bold tracking-tight text-slate-900`
   - Body & Prose: `text-xs text-slate-600 leading-relaxed`
   - Section Eyebrows: `text-xs font-bold uppercase tracking-wider text-cyan-700`
2. **Tabular Numerals & Telemetry Font:** `JetBrains Mono`, monospace
   - Probabilities: `font-mono font-bold text-slate-900 tabular-nums`
   - Coordinates & Measurements: `text-[11px] font-mono text-slate-500`
   - Entropy: `text-amber-800 font-mono font-bold`

---

## 4. Key Component Layouts

### 4.1 Top Navigation Bar (Strict 3-Zone Contract)
- **Height:** `h-16` (64px) sticky with backdrop blur (`backdrop-blur-md`).
- **Zone 1 (Left):** Wordmark `RetinaX` in display font (`text-lg font-bold text-slate-900 hover:text-cyan-600`).
- **Zone 2 (Center):** 3 clean navigation links (`Retinal Screener`, `Lesion AI Check`, `Safety & Real-World Tests`). Active link has `text-cyan-600 border-b-2 border-cyan-600 font-bold`.
- **Zone 3 (Right):** Primary action button: `Print Page` (`bg-slate-900 text-white rounded-lg px-3.5 py-1.5 text-xs font-semibold`).

### 4.2 Main Screener Stage (12-Column Grid)
- **Left Column (7 Cols):**
  - Layer Switcher (`Normal Photo` | `AI Highlights` | `Doctor's Lesions` | `Compare Both`).
  - 500x500 Canvas Viewport in natural dark ophthalmic bezel.
  - Live status telemetry bar (`Sharpness Variance`, `Illumination Uniformity`).
- **Right Column (5 Cols):**
  - Card 1: Image Quality Gate (`Clear & Gradable` vs `Photo Rejected: Blurry / Cataract Haze`).
  - Card 2: 5-Stage Disease Severity Grade with stage color chip and probability bars.
  - Card 3: Action & Specialist Referral (Confident Routine Care vs High Uncertainty Referral Slip).
  - Card 4 (Optional): Doctor Lesion Match (% Overlap, IoU, Pointing Game).
  - Expandable Panel: Measured Image Features & Mathematical Constants.
