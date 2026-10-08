from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime


class QualityHints(BaseModel):
    laplacian_variance: float
    illumination_uniformity: float
    fov_coverage: float


class QualityResult(BaseModel):
    status: Literal["GRADABLE", "UNGRADABLE", "UNAVAILABLE"]
    p_ungradable: float
    threshold: float
    hints: QualityHints
    recommended_action: Literal["NONE", "RECAPTURE_IMAGE"]


class ClassificationResult(BaseModel):
    predicted_stage: int
    stage_name: str
    raw_probabilities: List[float]
    calibrated_probabilities: List[float]


class UncertaintyResult(BaseModel):
    method: str
    temperature: float
    entropy_norm: float
    tau: float
    is_uncertain: bool
    mc_dropout: Optional[dict] = None


class ExplainabilityResult(BaseModel):
    method: Optional[str] = None
    target_class: Optional[int] = None
    heatmap_png_base64: Optional[str] = None


class ScreenRequest(BaseModel):
    eye: Literal["OD", "OS"] = "OD"
    xai_method: Literal["NONE", "GRAD_CAM_PP", "SCORE_CAM", "INTEGRATED_GRADIENTS"] = "NONE"
    mc_dropout: bool = False


class ScreenResponse(BaseModel):
    screening_id: str
    inference_mode: Literal["model", "quality_only"]
    model_provenance: Optional[str] = None
    quality: QualityResult
    classification: Optional[ClassificationResult] = None
    uncertainty: Optional[UncertaintyResult] = None
    clinical_action: Literal["ACCEPT_GRADE", "SPECIALIST_REFERRAL", "RECAPTURE_IMAGE", "MODEL_NOT_LOADED"]
    reason: Optional[str] = None
    explainability: Optional[ExplainabilityResult] = None


class HealthResponse(BaseModel):
    status: str
    device: str
    models: dict
    provenance: Optional[str] = None
    calibration_loaded: bool
    version: str


class QualityOnlyResponse(BaseModel):
    screening_id: str
    quality: QualityResult
    classification: Optional[ClassificationResult] = None
    uncertainty: Optional[UncertaintyResult] = None
    clinical_action: Literal["RECAPTURE_IMAGE"]
    reason: str = "UNGRADABLE"
    explainability: Optional[ExplainabilityResult] = None


class ReferralCreate(BaseModel):
    screening_id: str
    reason: str


class ReferralResponse(BaseModel):
    id: str
    screening_id: str
    reason: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class ScreeningSummary(BaseModel):
    id: str
    patient_id: Optional[str] = None
    eye: str
    quality_status: str
    predicted_stage: Optional[int] = None
    entropy_norm: Optional[float] = None
    clinical_action: str
    created_at: datetime

    class Config:
        from_attributes = True


class CohortImage(BaseModel):
    name: str
    label: int
    has_masks: bool
    image_url: str
    mask_urls: dict


class LesionValidationRequest(BaseModel):
    method: Literal["GRAD_CAM_PP", "SCORE_CAM", "INTEGRATED_GRADIENTS"] = "GRAD_CAM_PP"


class LesionValidationResult(BaseModel):
    image_name: str
    method: str
    target_class: int
    dice: dict
    iou: dict
    precision: dict
    recall: dict
    pointing_game: dict
    lesion_auroc: dict