import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation=True):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class LiverAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for liver CT anomaly detection.
    Trained exclusively on normal slices; elevated reconstruction
    error indicates a potential anomaly.
    Input:  (B, 1, 128, 128)  -- single-channel grayscale CT slice
    Output: (B, 1, 128, 128)  -- reconstructed slice
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            ConvBlock(1, 32),        # 64x64
            ConvBlock(32, 64),       # 32x32
            ConvBlock(64, 128),      # 16x16
            ConvBlock(128, 256),     # 8x8
        )

        self.bottleneck_enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim),
            nn.ReLU(inplace=True),
        )

        self.bottleneck_dec = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            ConvTransposeBlock(256, 128),   # 16x16
            ConvTransposeBlock(128, 64),    # 32x32
            ConvTransposeBlock(64, 32),     # 64x64
            ConvTransposeBlock(32, 1, activation=False),  # 128x128
            nn.Sigmoid(),
        )

    def encode(self, x):
        features = self.encoder(x)
        return self.bottleneck_enc(features)

    def decode(self, z):
        x = self.bottleneck_dec(z)
        x = x.view(-1, 256, 8, 8)
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


def load_model(weights_path: str = "model/weights/liver_ae_weights.pth", device: str = "cpu") -> LiverAutoencoder:
    model = LiverAutoencoder()
    try:
        state = torch.load(weights_path, map_location=torch.device(device))
        model.load_state_dict(state)
        model.eval()
        logger.info("Model weights loaded from %s", weights_path)
    except FileNotFoundError:
        logger.warning("Weights not found at %s — using untrained model.", weights_path)
        model.eval()
    return model
