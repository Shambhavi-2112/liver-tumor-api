import torch
from torchvision import transforms
from PIL import Image
import numpy as np


HU_LIVER_MIN = -100
HU_LIVER_MAX = 400


def hu_window_normalize(image_array: np.ndarray) -> np.ndarray:
    clipped = np.clip(image_array, HU_LIVER_MIN, HU_LIVER_MAX)
    return (clipped - HU_LIVER_MIN) / (HU_LIVER_MAX - HU_LIVER_MIN)


def get_inference_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def preprocess_pil_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL image to an inference-ready tensor of shape (1, 1, 128, 128).
    """
    transform = get_inference_transform()
    tensor = transform(image)
    return tensor.unsqueeze(0)
