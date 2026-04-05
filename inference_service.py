import torch
import torch.nn.functional as F
import logging
from model.model import LiverAutoencoder

logger = logging.getLogger(__name__)


class InferenceService:

    def __init__(self, model: LiverAutoencoder, threshold: float):
        self.model = model
        self.threshold = threshold

    @torch.inference_mode()
    def predict(self, input_tensor: torch.Tensor) -> dict:
        reconstructed = self.model(input_tensor)
        mse = F.mse_loss(reconstructed, input_tensor).item()
        is_anomaly = mse > self.threshold

        logger.info("Inference complete | MSE=%.6f | threshold=%.4f | anomaly=%s", mse, self.threshold, is_anomaly)

        return {
            "anomaly_score": round(mse, 6),
            "is_anomaly": is_anomaly,
            "threshold_used": self.threshold,
            "message": "Potential anomaly detected. Please consult a radiologist." if is_anomaly else "No anomaly detected.",
        }
