import os
import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

try:
    import nibabel as nib
except ImportError:
    raise SystemExit("Run: pip install nibabel tqdm")

# ── HU windowing ─────────────────────────────────────────────────────────────

def hu_window(arr: np.ndarray, center: int = 50, width: int = 350) -> np.ndarray:
    """
    Apply a liver-specific Hounsfield Unit window then normalise to [0, 1].
    Liver soft-tissue window: centre=50, width=350 → [-125, 225] HU.
    """
    lo = center - width // 2
    hi = center + width // 2
    clipped = np.clip(arr, lo, hi)
    return (clipped - lo) / (hi - lo)


def resize_slice(arr: np.ndarray, size: int = 128) -> np.ndarray:
    from PIL import Image
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


# ── Per-case processing ───────────────────────────────────────────────────────

def process_case(volume_path: str, seg_path: str, idx: int,
                 out_dir: Path, val_ratio: float = 0.10) -> dict:
    """
    Extract 2D slices from one CT volume.
    Returns counts for reporting.
    """
    vol = nib.load(volume_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()

    counts = {"normal": 0, "tumor": 0}
    normal_slices = []

    for z in range(vol.shape[2]):
        raw = vol[:, :, z]
        mask = seg[:, :, z]

        windowed = hu_window(raw)
        resized = resize_slice(windowed, size=128)

        if mask.sum() == 0:
            normal_slices.append((z, resized))
        else:
            save_path = out_dir / "val" / "tumor" / f"{idx}_{z}.npy"
            np.save(save_path, resized)
            counts["tumor"] += 1

    # Split normal slices into train/val
    random.shuffle(normal_slices)
    n_val = max(1, int(len(normal_slices) * val_ratio))
    val_slices = normal_slices[:n_val]
    train_slices = normal_slices[n_val:]

    for z, arr in train_slices:
        np.save(out_dir / "train" / "normal" / f"{idx}_{z}.npy", arr)
        counts["normal"] += 1

    for z, arr in val_slices:
        np.save(out_dir / "val" / "normal" / f"{idx}_{z}.npy", arr)

    return counts


# ── Visualisation helpers ─────────────────────────────────────────────────────

def make_sample_grid(npy_dir: Path, title: str, n: int = 12,
                     cmap: str = "gray", out_path: Path = None):
    files = sorted(npy_dir.glob("*.npy"))
    if not files:
        return
    chosen = random.sample(files, min(n, len(files)))
    cols = 4
    rows = (len(chosen) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.patch.set_facecolor("#0D1B2A")
    axes = axes.flatten()

    for i, f in enumerate(chosen):
        arr = np.load(f)
        axes[i].imshow(arr, cmap=cmap, vmin=0, vmax=1)
        axes[i].set_title(f.stem, fontsize=7, color="white")
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="#0D1B2A")
    plt.close()


def make_stats_plot(out_dir: Path, vis_dir: Path):
    normal_files = list((out_dir / "train" / "normal").glob("*.npy"))
    tumor_files = list((out_dir / "val" / "tumor").glob("*.npy"))

    def sample_means(files, k=200):
        chosen = random.sample(files, min(k, len(files)))
        return [np.load(f).mean() for f in chosen]

    normal_means = sample_means(normal_files)
    tumor_means = sample_means(tumor_files)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0D1B2A")

    for ax in axes:
        ax.set_facecolor("#0A1628")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1B6CA8")

    axes[0].hist(normal_means, bins=40, color="#06D6A0", alpha=0.8, label="Normal")
    axes[0].hist(tumor_means, bins=40, color="#F72585", alpha=0.8, label="Tumor")
    axes[0].set_title("Mean Intensity Distribution", color="white", fontweight="bold")
    axes[0].set_xlabel("Mean pixel value", color="white")
    axes[0].set_ylabel("Count", color="white")
    axes[0].legend(facecolor="#0D1B2A", labelcolor="white")

    axes[1].boxplot(
        [normal_means, tumor_means],
        labels=["Normal", "Tumor"],
        patch_artist=True,
        boxprops=dict(facecolor="#1B6CA8", color="white"),
        medianprops=dict(color="#00B4D8", linewidth=2),
        whiskerprops=dict(color="white"),
        capprops=dict(color="white"),
        flierprops=dict(markerfacecolor="#F72585", marker="o", markersize=4)
    )
    axes[1].set_title("Intensity Boxplot", color="white", fontweight="bold")
    axes[1].tick_params(axis="x", colors="white")

    fig.suptitle("LiTS Slice Statistics", color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(vis_dir / "slice_stats.png", dpi=120, bbox_inches="tight", facecolor="#0D1B2A")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="lits_raw")
    parser.add_argument("--out_dir", default="data/lits")
    parser.add_argument("--vis_dir", default="data/lits_preview")
    parser.add_argument("--val_ratio", type=float, default=0.10)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    vis_dir = Path(args.vis_dir)

    for split in ["train/normal", "val/normal", "val/tumor"]:
        (out_dir / split).mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    volumes = sorted(raw_dir.glob("volume-*.nii*"))
    if not volumes:
        raise FileNotFoundError(f"No volume-*.nii files found in {raw_dir}")

    total = {"normal": 0, "tumor": 0}
    for vol_path in tqdm(volumes, desc="Processing volumes"):
        idx = vol_path.stem.split("-")[1].split(".")[0]
        seg_path = raw_dir / vol_path.name.replace("volume", "segmentation")
        if not seg_path.exists():
            print(f"  [WARN] Missing seg for {vol_path.name}, skipping")
            continue
        counts = process_case(str(vol_path), str(seg_path), idx, out_dir, args.val_ratio)
        total["normal"] += counts["normal"]
        total["tumor"] += counts["tumor"]

    print(f"\nDone — Normal slices: {total['normal']}  |  Tumour slices: {total['tumor']}")

    # Visualisations
    make_sample_grid(out_dir / "train" / "normal", "Sample Normal Slices",
                     out_path=vis_dir / "grid_normal.png")
    make_sample_grid(out_dir / "val" / "tumor", "Sample Tumour Slices",
                     cmap="inferno", out_path=vis_dir / "grid_tumor.png")
    make_stats_plot(out_dir, vis_dir)

    print(f"Visualisations saved to: {vis_dir}/")


if __name__ == "__main__":
    main()
