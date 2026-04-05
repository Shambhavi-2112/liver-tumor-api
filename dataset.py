import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class NormalLiverSliceDataset(Dataset):
    EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    def __init__(self, root_dir: str, split: str = "train"):
        self.root = Path(root_dir) / split / "normal"
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset split not found: {self.root}")

        self.files = [
            p for p in sorted(self.root.iterdir())
            if p.suffix.lower() in self.EXTENSIONS
        ]
        if not self.files:
            raise RuntimeError(f"No images found in {self.root}")

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert("L")
        return self.transform(image)
