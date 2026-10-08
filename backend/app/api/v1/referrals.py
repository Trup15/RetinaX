from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from backend.app.schemas import ReferralCreate, ReferralResponse, ReferralResponse
from backend.app.db import get_db, Referral, Screening

router = APIRouter()


@router.post("/referrals", response_model=ReferralResponse)
async def create_referral(
    referral: ReferralCreate,
    db: Session = Depends(get_db),
):
    # Verify screening exists
    screening = db.query(Screening).filter(Screening.id == referral.screening_id).first()
    if not screening:
        raise HTTPException(404, {"code": "SCREENING_NOT_FOUND", "message": "Screening not found"})
    
    db_referral = Referral(
        screening_id=referral.screening_id,
        reason=referral.reason,
        status="PENDING",
    )
    db.add(db_referral)
    db.commit()
    db.refresh(db_referral)
    
    return db_referral


@router.get("/referrals", response_model=List[ReferralResponse])
async def list_referrals(
    status: str = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    query = db.query(Referral)
    if status:
        query = query.filter(Referral.status == status)
    query = query.order_by(Referral.created_at.desc())
    query = query.offset(offset).limit(limit)
    return query.all()


@router.get("/referrals/{referral_id}", response_model=ReferralResponse)
async def get_referral(
    referral_id: int,
    db: Session = Depends(get_db),
):
    referral = db.query(Referral).filter(Referral.id == referral_id).first()
    if not referral:
        raise HTTPException(404, {"code": "REFERRAL_NOT_FOUND", "message": "Referral not found"})
    return referral


@router.put("/referrals/{referral_id}/status")
async def update_referral_status(
    referral_id: int,
    status: str,
    db: Session = Depends(get_db),
):
    referral = db.query(Referral).filter(Referral.id == referral_id).first()
    if not referral:
        raise HTTPException(404, {"code": "REFERRAL_NOT_FOUND", "message": "Referral not found"})
    
    referral.status = status
    db.commit()
    return {"id": referral.id, "status": referral.status}