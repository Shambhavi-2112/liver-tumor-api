from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io
import logging

from app.services.preprocessing import preprocess_pil_image
from app.services.inference_service import InferenceService
from app.schemas.response import DetectionResponse, HealthResponse
from app.core.config import settings
from model.model import load_model

logger = logging.getLogger(__name__)
router = APIRouter()

_model = load_model(weights_path=settings.weights_path, device=settings.device)
_service = InferenceService(model=_model, threshold=settings.anomaly_threshold)

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/tiff"}


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="active", version=settings.app_version)


@router.post("/detect", response_model=DetectionResponse)
async def detect_anomaly(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Accepted: {ALLOWED_CONTENT_TYPES}"
        )

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception as exc:
        logger.error("Image decode failed: %s", exc)
        raise HTTPException(status_code=422, detail="Could not decode image.")

    input_tensor = preprocess_pil_image(image)
    result = _service.predict(input_tensor)

    return DetectionResponse(filename=file.filename, **result)
