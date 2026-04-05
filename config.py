from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Liver Tumor Anomaly Detection API"
    app_version: str = "1.0.0"
    weights_path: str = "model/weights/liver_ae_weights.pth"
    anomaly_threshold: float = 0.05
    device: str = "cpu"
    log_level: str = "INFO"
    cors_origins: list[str] = ["*"]

    class Config:
        env_file = ".env"


settings = Settings()
