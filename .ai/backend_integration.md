# Backend Integration & API Bridge Guide — RetinaX

This document explains exactly how to connect the React frontend to the FastAPI PyTorch backend.

---

## 1. Environment Configuration

### Frontend (`frontend/.env.development` or `.env`):
```env
VITE_API_URL=http://localhost:8000
VITE_ENABLE_LOCAL_BACKEND=true
```

### Backend (`backend/.env`):
```env
APP_NAME=RetinaX-API
PORT=8000
DEBUG=True
CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
MODEL_DIR=./models_weights
SQLITE_DB_URL=sqlite:///./retinax.db
```

---

## 2. Frontend API Client (`frontend/src/utils/apiClient.ts`)

Create this file in the frontend to seamlessly call the FastAPI server:

```typescript
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface ScreeningResponse {
  screening_id: string;
  quality: {
    status: 'GRADABLE' | 'UNGRADABLE';
    gradable_probability: number;
    ungradable_probability: number;
    reason?: string;
    reason_label?: string;
  };
  classification: {
    predicted_stage: 0 | 1 | 2 | 3 | 4;
    calibrated_confidence: number;
    raw_probabilities: number[];
    calibrated_probabilities: number[];
  };
  uncertainty: {
    temperature: number;
    predictive_entropy: number;
    is_uncertain: boolean;
    mc_dropout_variance: number;
    clinical_action: 'ACCEPT_DIAGNOSIS' | 'SPECIALIST_REFERRAL';
  };
  explainability: {
    method: 'GRAD_CAM_PP' | 'SCORE_CAM' | 'INTEGRATED_GRADIENTS';
    heatmap_data_url: string;
    lesion_overlap?: {
      dice: number;
      iou: number;
      pointing_game_hit: boolean;
      recall: number;
    };
  };
}

/**
 * Sends image to Python FastAPI backend for full ML pipeline screening
 */
export async function sendScreeningRequest(
  imageBlob: Blob,
  eye: 'OD' | 'OS' = 'OD',
  xaiMethod: 'GRAD_CAM_PP' | 'SCORE_CAM' | 'INTEGRATED_GRADIENTS' = 'GRAD_CAM_PP'
): Promise<ScreeningResponse> {
  const formData = new FormData();
  formData.append('file', imageBlob, 'fundus.jpg');
  formData.append('eye', eye);
  formData.append('xai_method', xaiMethod);

  const res = await fetch(`${BASE_URL}/api/v1/screen`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail?.message || `Inference error: HTTP ${res.status}`);
  }

  return res.json();
}
```

---

## 3. Minimal FastAPI Skeleton (`backend/app/main.py`)

```python
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
import io

app = FastAPI(title="RetinaX Backend API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models_loaded": {
            "quality_gate": True,
            "dr_classifier": True,
        }
    }

@app.post("/api/v1/screen")
async def run_screening(
    file: UploadFile = File(...),
    eye: str = Form("OD"),
    xai_method: str = Form("GRAD_CAM_PP"),
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image.")

        # 1. FOV Crop
        # 2. Quality Gate (MobileNetV3)
        # 3. DR Classifier (EfficientNet-B3)
        # 4. Temperature Scaling (T=1.38) & Entropy
        # 5. Grad-CAM++ Generator
        
        # Return structured JSON matching frontend interface
        return {
            "screening_id": "scr_live_01",
            "quality": {
                "status": "GRADABLE",
                "gradable_probability": 0.965,
                "ungradable_probability": 0.035,
            },
            "classification": {
                "predicted_stage": 2,
                "calibrated_confidence": 0.784,
                "raw_probabilities": [0.03, 0.12, 0.78, 0.05, 0.02],
                "calibrated_probabilities": [0.05, 0.14, 0.72, 0.06, 0.03],
            },
            "uncertainty": {
                "temperature": 1.38,
                "predictive_entropy": 0.642,
                "is_uncertain": False,
                "mc_dropout_variance": 0.0034,
                "clinical_action": "ACCEPT_DIAGNOSIS",
            },
            "explainability": {
                "method": xai_method,
                "heatmap_data_url": "",
                "lesion_overlap": {
                    "dice": 0.542,
                    "iou": 0.371,
                    "pointing_game_hit": True,
                    "recall": 0.694,
                },
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={"code": "INFERENCE_FAILED", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

---

## 4. Running the Development Monorepo

```bash
# Terminal 1: Launch FastAPI Backend
cd backend
python -m venv venv
source venv/bin/activate   # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m app.main

# Terminal 2: Launch Vite React Frontend
cd frontend
npm install
npm run dev
```

Visit `http://localhost:3000` to interact with the full-stack system.
