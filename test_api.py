import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_png_bytes(width=128, height=128) -> bytes:
    img = Image.new("L", (width, height), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert "version" in data


def test_detect_valid_image():
    png_bytes = _make_png_bytes()
    response = client.post(
        "/api/v1/detect",
        files={"file": ("test_slice.png", png_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "anomaly_score" in data
    assert "is_anomaly" in data
    assert "threshold_used" in data
    assert isinstance(data["anomaly_score"], float)
    assert isinstance(data["is_anomaly"], bool)


def test_detect_invalid_file_type():
    response = client.post(
        "/api/v1/detect",
        files={"file": ("report.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert response.status_code == 400


def test_detect_corrupted_image():
    response = client.post(
        "/api/v1/detect",
        files={"file": ("broken.png", b"not_an_image", "image/png")},
    )
    assert response.status_code == 422
