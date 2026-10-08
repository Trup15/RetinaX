from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from sqlalchemy.orm import Session
import uuid
import json
import base64
import cv2
import numpy as np

from backend.app.schemas import ScreenRequest, ScreenResponse
from backend.app.db import get_db, Screening, Patient, Referral

router = APIRouter()


@router.post("/screen", response_model=ScreenResponse)
async def screen(
    file: UploadFile = File(...),
    eye: str = Form("OD"),
    xai_method: str = Form("NONE"),
    mc_dropout: bool = Form(False),
    db: Session = Depends(get_db),
    request: Request = None,
):
    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(415, {"code": "INVALID_IMAGE", "message": "File must be an image"})
    
    content = await file.read()
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(413, {"code": "FILE_TOO_LARGE", "message": "File size exceeds 15 MB"})
    
    # Decode image
    try:
        img_array = np.frombuffer(content, np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(400, {"code": "INVALID_IMAGE", "message": "Failed to decode image"})
    except Exception as e:
        raise HTTPException(400, {"code": "INVALID_IMAGE", "message": str(e)})
    
    # Run pipeline
    pipe = request.app.state.pipe
    result = pipe.screen(
        img_bgr,
        xai_method=xai_method if xai_method != "NONE" else None,
        mc_dropout=mc_dropout,
    )
    
    # Save to database
    screening_id = result["screening_id"]
    
    # Create patient if needed (simplified - no patient lookup for now)
    patient = None
    if result.get("model_provenance") and result["model_provenance"] != "quality_only":
        # Create a default patient for tracking
        patient = Patient(external_id=str(uuid.uuid4()))
        db.add(patient)
        db.flush()
    
    screening_record = Screening(
        id=screening_id,
        patient_id=patient.id if patient else None,
        eye=eye,
        quality_status=result["quality"]["status"],
        p_ungradable=result["quality"]["p_ungradable"],
        quality_threshold=result["quality"]["threshold"],
        predicted_stage=result["classification"]["predicted_stage"] if result.get("classification") else None,
        stage_name=result["classification"]["stage_name"] if result.get("classification") else None,
        raw_probabilities=json.dumps(result["classification"]["raw_probabilities"]) if result.get("classification") else None,
        calibrated_probabilities=json.dumps(result["classification"]["calibrated_probabilities"]) if result.get("classification") else None,
        entropy_norm=result["uncertainty"]["entropy_norm"] if result.get("uncertainty") else None,
        tau=result["uncertainty"]["tau"] if result.get("uncertainty") else None,
        is_uncertain=result["uncertainty"]["is_uncertain"] if result.get("uncertainty") else None,
        clinical_action=result["clinical_action"],
        reason=result["reason"],
        model_provenance=result.get("model_provenance"),
        heatmap_base64=result["explainability"]["heatmap_png_base64"] if result.get("explainability") else None,
        xai_method=result["explainability"]["method"] if result.get("explainability") else None,
    )
    db.add(screening_record)
    
    # Create referral if needed
    if result["clinical_action"] == "SPECIALIST_REFERRAL":
        referral = Referral(
            screening_id=screening_id,
            reason=result["reason"] or "HIGH_UNCERTAINTY",
            status="PENDING",
        )
        db.add(referral)
    
    db.commit()
    
    return ScreenResponse(**result)


@router.post("/quality")
async def quality_only(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    request: Request = None,
):
    """Quality gate only endpoint."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(415, {"code": "INVALID_IMAGE", "message": "File must be an image"})
    
    content = await file.read()
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(413, {"code": "FILE_TOO_LARGE", "message": "File size exceeds 15 MB"})
    
    try:
        img_array = np.frombuffer(content, np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(400, {"code": "INVALID_IMAGE", "message": "Failed to decode image"})
    except Exception as e:
        raise HTTPException(400, {"code": "INVALID_IMAGE", "message": str(e)})
    
    pipe = request.app.state.pipe
    result = pipe.screen(img_bgr, xai_method=None)
    
    # Only return quality gate result
    return {
        "screening_id": str(uuid.uuid4()),
        "quality": result["quality"],
        "classification": None,
        "uncertainty": None,
        "clinical_action": "RECAPTURE_IMAGE" if result["quality"]["status"] == "UNGRADABLE" else "UNKNOWN",
        "reason": "UNGRADABLE" if result["quality"]["status"] == "UNGRADABLE" else None,
    }