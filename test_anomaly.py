import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from model.model import LiverAutoencoder

# 1. Setup
device = torch.device("cpu")
model = LiverAutoencoder(latent_dim=256)
model.load_state_dict(torch.load("model/weights/liver_ae_weights.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def get_mse(img_path, add_tumor=False):
    img = Image.open(img_path).convert("L")
    if add_tumor:
        draw = ImageDraw.Draw(img)
        # Add a bright "tumor" spot
        draw.ellipse([60, 60, 75, 75], fill=255)
    
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(tensor)
    return torch.nn.functional.mse_loss(tensor, recon).item()

# 2. Compare
normal_path = "data/val/normal/synthetic_normal_0000.png"
score_normal = get_mse(normal_path, add_tumor=False)
score_anomaly = get_mse(normal_path, add_tumor=True)

print(f"Normal Score: {score_normal:.6f}")
print(f"Anomaly Score: {score_anomaly:.6f}")
print(f"Ratio: {score_anomaly/score_normal:.2f}x increase")