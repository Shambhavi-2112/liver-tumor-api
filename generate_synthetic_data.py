import os
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from pathlib import Path

def create_synthetic_healthy_scan(size=(128, 128)):
    img = Image.new('L', size, color=np.random.randint(10, 30))
    draw = ImageDraw.Draw(img)
    
    x0, y0 = np.random.randint(15, 35), np.random.randint(15, 35)
    x1, y1 = np.random.randint(90, 115), np.random.randint(90, 115)
    
    draw.ellipse([x0, y0, x1, y1], fill=np.random.randint(100, 160))
    
    img = img.filter(ImageFilter.GaussianBlur(radius=np.random.randint(4, 8)))
    
    img_array = np.array(img, dtype=np.float32)
    noise = np.random.normal(loc=0, scale=3.0, size=size)
    img_array = np.clip(img_array + noise, 0, 255)
    
    return Image.fromarray(img_array.astype(np.uint8))

def generate_dataset(root_dir="data", train_size=20, val_size=5):
    root_path = Path(root_dir)
    splits = {"train": train_size, "val": val_size}
    
    for split, count in splits.items():
        target_dir = root_path / split / "normal"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {count} {split} images in {target_dir}...")
        
        for i in range(count):
            img = create_synthetic_healthy_scan()
            file_path = target_dir / f"synthetic_normal_{i:04d}.png"
            img.save(file_path)
            
    print("\nSynthetic dataset generation complete!")

if __name__ == "__main__":
    generate_dataset(root_dir="data", train_size=20, val_size=5)