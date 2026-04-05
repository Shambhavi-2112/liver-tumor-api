import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class NormalLiverSliceDataset(Dataset):

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    NPY_EXT  = ".npy"

    def __init__(self, root_dir: str, split: str = "train", augment: bool = True):
        self.root = Path(root_dir) / split / "normal"
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset split not found: {self.root}")

        self.files = sorted(
            p for p in self.root.iterdir()
            if p.suffix.lower() in (self.IMG_EXTS | {self.NPY_EXT})
        )
        if not self.files:
            raise RuntimeError(f"No images found in {self.root}")

        self.augment = augment
        self._img_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self._aug = transforms.RandomHorizontalFlip(p=0.5)

    def __len__(self):
        return len(self.files)

    def _load(self, path: Path) -> torch.Tensor:
        if path.suffix.lower() == self.NPY_EXT:
            arr = np.load(path).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis]
            tensor = torch.from_numpy(arr)
            tensor = transforms.functional.resize(tensor, [128, 128])
            return (tensor - 0.5) / 0.5
        img = Image.open(path).convert("L")
        return self._img_transform(img)

    def __getitem__(self, idx: int) -> torch.Tensor:
        tensor = self._load(self.files[idx])
        if self.augment:
            tensor = self._aug(tensor)
        return tensor


class AnomalyEvalDataset(Dataset):

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}

    def __init__(self, root_dir: str):
        root = Path(root_dir) / "val"
        self.samples = []
        for p in sorted((root / "normal").iterdir()):
            if p.suffix.lower() in self.IMG_EXTS:
                self.samples.append((p, 0))
        for p in sorted((root / "tumor").iterdir()):
            if p.suffix.lower() in self.IMG_EXTS:
                self.samples.append((p, 1))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def _load(self, path: Path) -> torch.Tensor:
        if path.suffix.lower() == ".npy":
            arr = np.load(path).astype(np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis]
            tensor = torch.from_numpy(arr)
            tensor = transforms.functional.resize(tensor, [128, 128])
            return (tensor - 0.5) / 0.5
        img = Image.open(path).convert("L")
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])(img)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        return self._load(path), label
