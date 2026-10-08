from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
from pathlib import Path
import json
import base64
import cv2
import numpy as np

from backend.app.schemas import CohortImage, LesionValidationRequest, LesionValidationResult
from backend.app.db import get_db

router = APIRouter()


@router.get("/cohorts/idrid", response_model=List[CohortImage])
async def list_idrid_cohort():
    """List IDRiD sample images that have lesion masks."""
    meta_path = Path("metadata/idrid_meta.csv")
    if not meta_path.exists():
        return []
    
    import pandas as pd
    df = pd.read_csv(meta_path)
    # Filter to images with masks
    df = df[df["has_masks"] == 1]
    
    images = []
    for _, row in df.iterrows():
        mask_urls = {}
        for lesion in ["ma", "he", "ex", "se"]:
            col = f"{lesion}_mask"
            if row.get(col) and row[col]:
                mask_urls[lesion] = f"/api/v1/cohorts/idrid/mask/{row[col]}"
        
        images.append(CohortImage(
            name=Path(row["image_path"]).name,
            label=int(row["label"]),
            has_masks=True,
            image_url=f"/api/v1/cohorts/idrid/image/{Path(row['image_path']).name}",
            mask_urls=mask_urls,
        ))
    
    return images


@router.get("/cohorts/idrid/image/{image_name}")
async def get_idrid_image(image_name: str):
    """Serve IDRiD image file."""
    import pandas as pd
    from pathlib import Path as P
    
    # Find the image path
    meta_path = Path("metadata/idrid_meta.csv")
    if not meta_path.exists():
        raise HTTPException(404, "Image not found")
    
    import pandas as pd
    df = pd.read_csv(meta_path)
    row = df[df["image_path"].str.endswith(image_name)]
    if row.empty:
        raise HTTPException(404, "Image not found")
    
    img_path = P("smoke_data") / row.iloc[0]["image_path"]
    if not img_path.exists():
        raise HTTPException(404, "Image file not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(img_path)


@router.get("/cohorts/idrid/mask/{mask_path:path}")
async def get_idrid_mask(mask_path: str):
    """Serve IDRiD lesion mask file."""
    from pathlib import Path as P
    from fastapi.responses import FileResponse
    
    mask_file = P("smoke_data") / mask_path
    if not mask_file.exists():
        raise HTTPException(404, "Mask not found")
    
    return FileResponse(mask_file)


@router.get("/cohorts/idrid/{image_name}/lesion-validation", response_model=LesionValidationResult)
async def get_lesion_validation(
    image_name: str,
    method: str = Query("GRAD_CAM_PP"),
):
    """Get lesion validation metrics for a specific IDRiD image with masks."""
    # Load precomputed validation results
    results_path = Path("outputs/tables/xai_lesion.csv")
    if not results_path.exists():
        raise HTTPException(404, "Validation results not found. Run lesion validation first.")
    
    import pandas as pd
    df = pd.read_csv(results_path)
    
    # Filter by image and method
    df = df[df["image"] == image_name]
    if df.empty:
        raise HTTPException(404, "No validation data for this image")
    
    df = df[df["method"] == method.upper()]
    if df.empty:
        raise HTTPException(404, "No validation data for this method")
    
    # Aggregate metrics per lesion type
    lesions = ["ma", "he", "ex", "se", "union"]
    result = {
        "image_name": image_name,
        "method": method,
        "target_class": int(df.iloc[0]["target_class"]),
        "dice": {},
        "iou": {},
        "precision": {},
        "recall": {},
        "pointing_game": {},
        "lesion_auroc": {},
    }
    
    for lesion in lesions:
        lesion_rows = df[df["lesion_type"] == lesion]
        if not lesion_rows.empty:
            row = lesion_rows.iloc[0]
            result["dice"][lesion] = float(row.get("dice", 0))
            result["iou"][lesion] = float(row.get("iou", 0))
            result["precision"][lesion] = float(row.get("precision", 0))
            result["recall"][lesion] = float(row.get("recall", 0))
            result["pointing_game"][lesion] = bool(row.get("pointing_game", False))
            result["lesion_auroc"][lesion] = float(row.get("lesion_auroc", 0))
    
    return result