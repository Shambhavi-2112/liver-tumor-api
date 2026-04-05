from pydantic import BaseModel, Field


class DetectionResponse(BaseModel):
    filename: str
    anomaly_score: float = Field(..., description="Mean Squared Error between input and reconstruction")
    is_anomaly: bool = Field(..., description="True if anomaly_score exceeds configured threshold")
    threshold_used: float
    message: str

    model_config = {"json_schema_extra": {"example": {
        "filename": "slice_042.png",
        "anomaly_score": 0.0312,
        "is_anomaly": False,
        "threshold_used": 0.05,
        "message": "No anomaly detected."
    }}}


class HealthResponse(BaseModel):
    status: str
    version: str
