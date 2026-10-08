from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from retinax.pipeline import RetinaXPipeline
from retinax.config import load_config
from backend.app.api.v1 import screening, referrals, results, cohorts, quality
from backend.app.db import init_db

log = logging.getLogger("retinax.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    app.state.pipe = RetinaXPipeline.load(cfg)
    if not app.state.pipe.dr_loaded:
        log.warning("[WARN] DR model not loaded; grading disabled.")
    # Initialize database
    init_db()
    yield


app = FastAPI(title="RetinaX API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(screening.router, prefix="/api/v1")
app.include_router(quality.router, prefix="/api/v1")
app.include_router(referrals.router, prefix="/api/v1")
app.include_router(results.router, prefix="/api/v1")
app.include_router(cohorts.router, prefix="/api/v1")


@app.get("/api/v1/health")
async def health():
    pipe = app.state.pipe
    return {
        "status": "healthy",
        "device": str(pipe.device) if hasattr(pipe, 'device') else "unknown",
        "models": {
            "quality": pipe.quality_loaded,
            "dr": pipe.dr_loaded,
        },
        "provenance": pipe.quality_loaded and "quality_only" or (pipe.dr_loaded and "aptos_dev_fold0" or None),
        "calibration_loaded": pipe.calib_loaded,
        "version": "1.0.0",
    }