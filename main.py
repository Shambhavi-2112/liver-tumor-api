from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging
from app.routes.inference import router

setup_logging(settings.log_level)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Reconstruction-based anomaly detection for liver CT slices. "
        "Outputs an anomaly score derived from autoencoder reconstruction error. "
        "This system is a research prototype and must not be used as a standalone diagnostic tool."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
