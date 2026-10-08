from fastapi import APIRouter, UploadFile, File, HTTPException, Request
import cv2
import numpy as np

from backend.app.schemas import ScreenResponse

router = APIRouter()


@router.post("/quality")
async def quality_only(
    file: UploadFile = File(...),
    request: Request = None,
):
    """Quality gate only endpoint - runs quality model without DR classification."""
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
    
    # Run quality gate
    pipe = request.app.state.pipe
    result = pipe.screen(img_bgr, xai_method=None)
    
    return {
        "screening_id": result["screening_id"],
        "quality": result["quality"],
        "classification": None,
        "uncertainty": None,
        "clinical_action": "RECAPTURE_IMAGE" if result["quality"]["status"] == "UNGRADABLE" else "UNKNOWN",
        "reason": "UNGRADABLE" if result["quality"]["status"] == "UNGRADABLE" else None,
        "explainability": None,
    }