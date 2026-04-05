import sys
import os
import yaml
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.model import LiverAutoencoder
from training.dataset import NormalLiverSliceDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def ssim_loss(pred, target, window_size=11):
    mu_x = F.avg_pool2d(pred, window_size, 1, window_size // 2)
    mu_y = F.avg_pool2d(target, window_size, 1, window_size // 2)
    mu_x_sq, mu_y_sq = mu_x ** 2, mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x = F.avg_pool2d(pred ** 2, window_size, 1, window_size // 2) - mu_x_sq
    sigma_y = F.avg_pool2d(target ** 2, window_size, 1, window_size // 2) - mu_y_sq
    sigma_xy = F.avg_pool2d(pred * target, window_size, 1, window_size // 2) - mu_xy
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2))
    return 1 - ssim_map.mean()


def combined_loss(pred, target, alpha: float = 0.7):
    """alpha * MSE + (1 - alpha) * (1 - SSIM)"""
    return alpha * F.mse_loss(pred, target) + (1 - alpha) * ssim_loss(pred, target)


def train(config: dict):
    device = torch.device(config.get("device", "cpu"))
    logger.info("Training on device: %s", device)

    train_dataset = NormalLiverSliceDataset(config["data_root"], split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 2),
        pin_memory=(device.type == "cuda"),
    )

    model = LiverAutoencoder(latent_dim=config["latent_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

    best_loss = float("inf")
    output_path = Path(config["weights_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config["num_epochs"] + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = combined_loss(reconstructed, batch, alpha=config.get("loss_alpha", 0.7))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        logger.info("Epoch %d/%d — Loss: %.6f", epoch, config["num_epochs"], avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_path)
            logger.info("Checkpoint saved (best_loss=%.6f)", best_loss)

    logger.info("Training complete. Best loss: %.6f", best_loss)


if __name__ == "__main__":
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
