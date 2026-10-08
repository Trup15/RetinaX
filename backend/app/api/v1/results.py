from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
import json
from pathlib import Path

from backend.app.schemas import ScreeningSummary
from backend.app.db import get_db, Screening

router = APIRouter()


@router.get("/results")
async def get_results():
    """Serve experiment summary from outputs/tables/summary.json"""
    summary_path = Path("outputs/tables/summary.json")
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {"available": False}


@router.get("/screenings", response_model=List[ScreeningSummary])
async def list_screenings(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """List all screenings with summary info."""
    screenings = db.query(Screening).order_by(Screening.created_at.desc()).offset(offset).limit(limit).all()
    return [
        ScreeningSummary(
            id=s.id,
            patient_id=s.patient_id,
            eye=s.eye,
            quality_status=s.quality_status,
            predicted_stage=s.predicted_stage,
            entropy_norm=s.entropy_norm,
            clinical_action=s.clinical_action,
            created_at=s.created_at,
        )
        for s in screenings
    ]


@router.get("/screenings/{screening_id}")
async def get_screening(
    screening_id: int,
    db: Session = Depends(get_db),
):
    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    if not screening:
        raise HTTPException(404, {"code": "SCREENING_NOT_FOUND", "message": "Screening not found"})
    
    return {
        "id": screening.id,
        "patient_id": screening.patient_id,
        "eye": screening.eye,
        "quality_status": screening.quality_status,
        "p_ungradable": screening.p_ungradable,
        "quality_threshold": screening.quality_threshold,
        "predicted_stage": screening.predicted_stage,
        "stage_name": screening.stage_name,
        "raw_probabilities": json.loads(screening.raw_probabilities) if screening.raw_probabilities else None,
        "calibrated_probabilities": json.loads(screening.calibrated_probabilities) if screening.calibrated_probabilities else None,
        "entropy_norm": screening.entropy_norm,
        "tau": screening.tau,
        "is_uncertain": screening.is_uncertain,
        "clinical_action": screening.clinical_action,
        "reason": screening.reason,
        "model_provenance": screening.model_provenance,
        "heatmap_base64": screening.heatmap_base64,
        "xai_method": screening.xai_method,
        "created_at": screening.created_at,
    }