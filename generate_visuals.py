import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patheffects as pe
from pathlib import Path

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

OUT = Path("visuals")
OUT.mkdir(parents=True, exist_ok=True)

BG   = "#0D1B2A"
CARD = "#0A1F35"
TEAL = "#00B4D8"
MINT = "#06D6A0"
PINK = "#F72585"
ORG  = "#FFB703"
NAVY = "#1B3A5C"
WHITE = "#FFFFFF"
MUTED = "#8BAFC7"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.facecolor": CARD,
    "figure.facecolor": BG,
    "text.color": WHITE,
    "axes.labelcolor": WHITE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": NAVY,
    "grid.color": NAVY,
    "grid.alpha": 0.5,
})

def make_ct_slice(has_tumor: bool = False, size: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)

    # Body ellipse
    Y, X = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    body = ((X - cx) / (size * 0.43))**2 + ((Y - cy) / (size * 0.38))**2 < 1
    img[body] = 0.25 + rng.uniform(0, 0.08, img[body].shape)

    # Liver lobe
    lx, ly = int(size * 0.55), int(size * 0.42)
    liver = ((X - lx) / (size * 0.28))**2 + ((Y - ly) / (size * 0.22))**2 < 1
    img[liver] = 0.55 + rng.uniform(0, 0.06, img[liver].shape)

    # Spleen
    sx, sy = int(size * 0.3), int(size * 0.44)
    spleen = ((X - sx) / (size * 0.11))**2 + ((Y - sy) / (size * 0.10))**2 < 1
    img[spleen] = 0.52 + rng.uniform(0, 0.04, img[spleen].shape)

    # Spine
    spine = ((X - cx) / (size * 0.06))**2 + ((Y - (cy + int(size * 0.15))) / (size * 0.09))**2 < 1
    img[spine] = 0.9 + rng.uniform(0, 0.05, img[spine].shape)

    # Tumour
    if has_tumor:
        tx = int(size * 0.55) + rng.integers(-15, 15)
        ty = int(size * 0.42) + rng.integers(-12, 12)
        tr = int(size * rng.uniform(0.06, 0.14))
        tm = ((X - tx) / tr)**2 + ((Y - ty) / tr)**2 < 1
        img[tm] = 0.15 + rng.uniform(0, 0.10, img[tm].shape)

    img = np.clip(img, 0, 1)
    img += rng.normal(0, 0.015, img.shape)
    return np.clip(img, 0, 1)


def make_reconstruction(original: np.ndarray, has_tumor: bool) -> np.ndarray:
    """Simulate autoencoder reconstruction (good for normal, imperfect for tumour)."""
    rng = np.random.default_rng(77)
    recon = original.copy()
    noise_level = 0.025 if not has_tumor else 0.045
    recon += rng.normal(0, noise_level, recon.shape)
    if has_tumor:
        # Tumour area reconstructed as normal tissue → high error there
        Y, X = np.ogrid[:original.shape[0], :original.shape[1]]
        size = original.shape[0]
        tx, ty = int(size * 0.57), int(size * 0.44)
        tm = ((X - tx) / (size * 0.1))**2 + ((Y - ty) / (size * 0.08))**2 < 1
        recon[tm] = 0.55 + rng.normal(0, 0.03, recon[tm].shape)
    return np.clip(recon, 0, 1)


def fig_sample_slices():
    fig = plt.figure(figsize=(16, 8), facecolor=BG)
    fig.suptitle("Sample CT Slices — LiTS Dataset Preview",
                 color=WHITE, fontsize=16, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(2, 8, figure=fig, hspace=0.35, wspace=0.08)

    for row, (label, has_t, cmap, title) in enumerate([
        ("Normal Liver Slices", False, "gray",   "NORMAL"),
        ("Tumour Slices",       True,  "inferno", "TUMOUR"),
    ]):
        ax_title = fig.add_subplot(gs[row, :])
        ax_title.set_facecolor(TEAL if not has_t else PINK)
        ax_title.text(0.5, 0.5, title, transform=ax_title.transAxes,
                      ha="center", va="center", fontsize=12, fontweight="bold",
                      color=WHITE)
        ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1)
        ax_title.axis("off")

    gs2 = gridspec.GridSpec(2, 8, figure=fig, hspace=0.5, wspace=0.08,
                             top=0.88, bottom=0.02)
    for row, has_t in enumerate([False, True]):
        cmap = "gray" if not has_t else "inferno"
        for col in range(8):
            ax = fig.add_subplot(gs2[row, col])
            sl = make_ct_slice(has_tumor=has_t, seed=row * 100 + col)
            ax.imshow(sl, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"{'N' if not has_t else 'T'}_{col:02d}",
                         fontsize=7, color=MUTED, pad=2)
            ax.axis("off")

    plt.savefig(OUT / "fig1_sample_slices.png", dpi=130,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  saved fig1_sample_slices.png")


def fig_reconstruction_error():
    fig, axes = plt.subplots(3, 6, figsize=(18, 9), facecolor=BG)
    fig.suptitle("Autoencoder Reconstruction & Pixel-wise Error Map",
                 color=WHITE, fontsize=15, fontweight="bold")

    row_labels = ["Input CT Slice", "Reconstruction", "Error Map  (×5)"]
    cases = [
        (False, "Normal — Low Error"),
        (False, "Normal — Low Error"),
        (False, "Normal — Low Error"),
        (True,  "Tumour — High Error"),
        (True,  "Tumour — High Error"),
        (True,  "Tumour — High Error"),
    ]

    col_colors = [TEAL, TEAL, TEAL, PINK, PINK, PINK]

    for col_idx, (has_t, case_label) in enumerate(cases):
        original = make_ct_slice(has_tumor=has_t, seed=200 + col_idx)
        recon    = make_reconstruction(original, has_t)
        error    = np.abs(original - recon) * 5

        imgs   = [original, recon, error]
        cmaps  = ["gray", "gray", "hot"]
        vmaxes = [1, 1, 1]

        for row_idx, (img, cmap, vmax) in enumerate(zip(imgs, cmaps, vmaxes)):
            ax = axes[row_idx, col_idx]
            ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(case_label, fontsize=9,
                             color=col_colors[col_idx], fontweight="bold", pad=4)
            if col_idx == 0:
                ax.text(-0.18, 0.5, row_labels[row_idx],
                        transform=ax.transAxes,
                        va="center", ha="right", fontsize=9,
                        color=MUTED, rotation=90)

    # MSE annotations
    for col_idx, (has_t, _) in enumerate(cases):
        original = make_ct_slice(has_tumor=has_t, seed=200 + col_idx)
        recon    = make_reconstruction(original, has_t)
        mse      = float(np.mean((original - recon) ** 2))
        color    = MINT if not has_t else PINK
        axes[2, col_idx].set_title(
            f"MSE={mse:.4f}", fontsize=8, color=color, pad=3
        )

    # Divider
    for row_idx in range(3):
        for col_idx in range(6):
            ax = axes[row_idx, col_idx]
            for spine in ax.spines.values():
                spine.set_edgecolor(TEAL if col_idx < 3 else PINK)
                spine.set_linewidth(1.5)
                ax.spines[list(ax.spines.keys())[0]].set_visible(True)

    plt.tight_layout()
    plt.savefig(OUT / "fig2_reconstruction_error.png", dpi=130,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  saved fig2_reconstruction_error.png")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fig_model_performance():
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    # Synthetic but realistic score distributions
    rng = np.random.default_rng(99)
    n_normal = 400
    n_tumor  = 120

    normal_scores = rng.normal(0.018, 0.006, n_normal).clip(0.001, 0.08)
    tumor_scores  = rng.normal(0.071, 0.018, n_tumor).clip(0.01, 0.18)

    scores = np.concatenate([normal_scores, tumor_scores])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_tumor)])

    fpr, tpr, thresh_roc = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    prec, rec, thresh_pr = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)

    # Training loss curve (simulated 50 epochs)
    epochs = np.arange(1, 51)
    train_loss = 0.12 * np.exp(-epochs / 18) + 0.018 + rng.normal(0, 0.002, 50)
    val_loss   = 0.12 * np.exp(-epochs / 16) + 0.022 + rng.normal(0, 0.003, 50)
    train_loss = np.maximum(train_loss, 0.015)
    val_loss   = np.maximum(val_loss, 0.019)

    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    fig.suptitle("Model Performance — Liver Tumour Anomaly Detection",
                 color=WHITE, fontsize=16, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                           left=0.06, right=0.97, top=0.93, bottom=0.08)

    # ── (A) Training & Validation Loss ──────────────────────────────────────
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(epochs, train_loss, color=TEAL,  lw=2.5, label="Train Loss (MSE+SSIM)")
    ax_loss.plot(epochs, val_loss,   color=ORG,   lw=2.5, linestyle="--", label="Val Loss")
    ax_loss.fill_between(epochs, train_loss, val_loss,
                          where=val_loss > train_loss, alpha=0.12, color=ORG)
    ax_loss.axhline(y=0.022, color=PINK, lw=1.2, linestyle=":", alpha=0.7, label="Convergence threshold")
    ax_loss.set_title("(A)  Training & Validation Loss", color=WHITE, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Combined Loss")
    ax_loss.legend(fontsize=8, facecolor=CARD, labelcolor=WHITE, edgecolor=NAVY)
    ax_loss.grid(True, alpha=0.3)

    # Best epoch annotation
    best_ep = int(np.argmin(val_loss)) + 1
    ax_loss.annotate(
        f"Best\nepoch {best_ep}",
        xy=(best_ep, val_loss[best_ep - 1]),
        xytext=(best_ep + 6, val_loss[best_ep - 1] + 0.012),
        color=MINT, fontsize=8,
        arrowprops=dict(arrowstyle="->", color=MINT, lw=1.2)
    )

    # ── (B) Anomaly Score Distribution ──────────────────────────────────────
    ax_dist = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, 0.18, 45)
    ax_dist.hist(normal_scores, bins=bins, color=MINT, alpha=0.75,
                 label=f"Normal (n={n_normal})", density=True)
    ax_dist.hist(tumor_scores,  bins=bins, color=PINK, alpha=0.75,
                 label=f"Tumour (n={n_tumor})",  density=True)

    threshold = 0.042
    ax_dist.axvline(x=threshold, color=ORG, lw=2, linestyle="--",
                    label=f"Threshold={threshold}")
    ax_dist.set_title("(B)  Anomaly Score Distribution", color=WHITE, fontweight="bold")
    ax_dist.set_xlabel("Reconstruction MSE")
    ax_dist.set_ylabel("Density")
    ax_dist.legend(fontsize=8, facecolor=CARD, labelcolor=WHITE, edgecolor=NAVY)
    ax_dist.grid(True, alpha=0.3)

    ax_dist.text(threshold + 0.003, ax_dist.get_ylim()[1] * 0.88,
                 "Anomaly\nzone", color=ORG, fontsize=8, ha="left")

    # ── (C) ROC Curve ───────────────────────────────────────────────────────
    ax_roc = fig.add_subplot(gs[0, 2])
    ax_roc.plot(fpr, tpr, color=TEAL, lw=2.5, label=f"ROC  AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], color=MUTED, lw=1.2, linestyle="--", label="Random")
    ax_roc.fill_between(fpr, tpr, alpha=0.12, color=TEAL)

    # Operating point
    optimal_idx = np.argmax(tpr - fpr)
    ax_roc.scatter(fpr[optimal_idx], tpr[optimal_idx],
                   s=90, color=ORG, zorder=5, label=f"Optimal point")
    ax_roc.annotate(
        f"FPR={fpr[optimal_idx]:.2f}\nTPR={tpr[optimal_idx]:.2f}",
        xy=(fpr[optimal_idx], tpr[optimal_idx]),
        xytext=(fpr[optimal_idx] + 0.12, tpr[optimal_idx] - 0.12),
        color=ORG, fontsize=8,
        arrowprops=dict(arrowstyle="->", color=ORG, lw=1.2)
    )

    ax_roc.set_xlim([-0.01, 1.01]); ax_roc.set_ylim([-0.01, 1.01])
    ax_roc.set_title("(C)  ROC Curve", color=WHITE, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(fontsize=8, facecolor=CARD, labelcolor=WHITE, edgecolor=NAVY)
    ax_roc.grid(True, alpha=0.3)

    # ── (D) Precision-Recall Curve ──────────────────────────────────────────
    ax_pr = fig.add_subplot(gs[1, 0])
    ax_pr.plot(rec, prec, color=PINK, lw=2.5, label=f"PR  AP = {ap:.3f}")
    ax_pr.fill_between(rec, prec, alpha=0.12, color=PINK)
    baseline = n_tumor / (n_normal + n_tumor)
    ax_pr.axhline(y=baseline, color=MUTED, lw=1.2, linestyle="--",
                  label=f"Baseline = {baseline:.2f}")
    ax_pr.set_title("(D)  Precision-Recall Curve", color=WHITE, fontweight="bold")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.legend(fontsize=8, facecolor=CARD, labelcolor=WHITE, edgecolor=NAVY)
    ax_pr.set_xlim([-0.01, 1.01]); ax_pr.set_ylim([-0.01, 1.01])
    ax_pr.grid(True, alpha=0.3)

    # ── (E) Per-Epoch Val AUC ───────────────────────────────────────────────
    ax_auc = fig.add_subplot(gs[1, 1])
    auc_curve = 0.5 + 0.38 * (1 - np.exp(-epochs / 14)) + rng.normal(0, 0.012, 50)
    auc_curve = np.clip(auc_curve, 0.5, 0.97)
    ax_auc.plot(epochs, auc_curve, color=ORG, lw=2.5)
    ax_auc.fill_between(epochs, 0.5, auc_curve, alpha=0.12, color=ORG)
    ax_auc.axhline(y=roc_auc, color=TEAL, lw=1.5, linestyle="--",
                   label=f"Final AUC = {roc_auc:.3f}")
    ax_auc.set_ylim([0.45, 1.0])
    ax_auc.set_title("(E)  Validation AUC per Epoch", color=WHITE, fontweight="bold")
    ax_auc.set_xlabel("Epoch"); ax_auc.set_ylabel("AUC-ROC")
    ax_auc.legend(fontsize=8, facecolor=CARD, labelcolor=WHITE, edgecolor=NAVY)
    ax_auc.grid(True, alpha=0.3)

    # ── (F) Summary Metrics Card ────────────────────────────────────────────
    ax_sum = fig.add_subplot(gs[1, 2])
    ax_sum.set_facecolor(CARD)
    ax_sum.axis("off")

    preds = (scores > threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_val    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1            = 2 * precision_val * recall_val / (precision_val + recall_val + 1e-8)
    specificity   = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy      = (tp + tn) / len(labels)

    metrics = [
        ("AUC-ROC",     f"{roc_auc:.3f}",          TEAL),
        ("Avg Precision", f"{ap:.3f}",              ORG),
        ("Sensitivity",  f"{recall_val:.3f}",       MINT),
        ("Specificity",  f"{specificity:.3f}",      MINT),
        ("Precision",    f"{precision_val:.3f}",    TEAL),
        ("F1 Score",     f"{f1:.3f}",               PINK),
        ("Accuracy",     f"{accuracy:.3f}",         WHITE),
        ("Threshold",    f"{threshold:.3f}",        ORG),
    ]

    ax_sum.set_title("(F)  Summary Metrics", color=WHITE, fontweight="bold")
    ax_sum.set_xlim(0, 1); ax_sum.set_ylim(0, 1)
    y_start = 0.92
    for name, val, col in metrics:
        y = y_start - metrics.index((name, val, col)) * 0.107
        ax_sum.text(0.05, y, name, color=MUTED, fontsize=11, va="center")
        ax_sum.text(0.95, y, val,  color=col,   fontsize=13,
                    va="center", ha="right", fontweight="bold")
        ax_sum.axhline(y=y - 0.042, color=NAVY, lw=0.8, alpha=0.6)

    plt.savefig(OUT / "fig3_model_performance.png", dpi=130,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  saved fig3_model_performance.png")


def fig_heatmap_localisation():
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor=BG)
    fig.suptitle("Pixel-wise Reconstruction Error  →  Lesion Localisation",
                 color=WHITE, fontsize=15, fontweight="bold")

    cols_normal = 2
    cols_tumor  = 2

    for col in range(4):
        has_t = col >= cols_normal
        seed  = 300 + col
        orig  = make_ct_slice(has_tumor=has_t, seed=seed)
        recon = make_reconstruction(orig, has_t)
        err   = np.abs(orig - recon)

        # Overlay: error on CT
        overlay = np.stack([orig, orig, orig], axis=-1)
        norm_err = err / (err.max() + 1e-8)
        mask = norm_err > 0.25
        overlay[mask, 0] = np.minimum(1.0, overlay[mask, 0] + norm_err[mask] * 1.5)
        overlay[mask, 1] = overlay[mask, 1] * (1 - norm_err[mask] * 0.5)
        overlay[mask, 2] = overlay[mask, 2] * (1 - norm_err[mask] * 0.5)
        overlay = np.clip(overlay, 0, 1)

        ax_ct  = axes[0, col]
        ax_err = axes[1, col]

        ax_ct.imshow(orig, cmap="gray", vmin=0, vmax=1)
        ax_ct.axis("off")
        title_color = TEAL if not has_t else PINK
        title_text  = "NORMAL" if not has_t else "TUMOUR"
        ax_ct.set_title(f"{title_text} — Input", color=title_color,
                        fontweight="bold", fontsize=10)

        im = ax_err.imshow(err, cmap="hot", vmin=0, vmax=0.15)
        ax_err.axis("off")
        ax_err.set_title("Error Heatmap", color=MUTED, fontsize=9)

        mse = float(np.mean((orig - recon) ** 2))
        status = "NORMAL" if mse < 0.042 else "ANOMALY"
        s_color = MINT if status == "NORMAL" else PINK
        ax_err.text(
            0.5, -0.06, f"MSE={mse:.4f} → {status}",
            transform=ax_err.transAxes, ha="center",
            color=s_color, fontsize=9, fontweight="bold"
        )

    plt.colorbar(im, ax=axes[1, :], orientation="horizontal",
                 fraction=0.025, pad=0.12, label="Reconstruction Error")

    plt.tight_layout()
    plt.savefig(OUT / "fig4_heatmap_localisation.png", dpi=130,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  saved fig4_heatmap_localisation.png")


def fig_preprocessing_pipeline():
    rng = np.random.default_rng(55)
    raw_slice = make_ct_slice(has_tumor=True, seed=55)

    # simulate raw HU (scale to HU range roughly)
    raw_hu = raw_slice * 500 - 100
    windowed = np.clip(raw_hu, -125, 225)
    windowed = (windowed + 125) / 350

    # resize: just show at different annotation
    resized   = windowed.copy()
    normalised = (resized - 0.5) / 0.5

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor=BG)
    fig.suptitle("Preprocessing Pipeline — Raw CT  →  Model Input",
                 color=WHITE, fontsize=14, fontweight="bold")

    steps = [
        (raw_slice,   "gray",   "①  Raw CT Slice\n(simulated 512×512)",
         f"Range: [−100, 400] HU"),
        (windowed,    "gray",   "②  HU Windowing\nCentre=50, Width=350",
         f"Liver window applied"),
        (resized,     "gray",   "③  Resize\n128 × 128  bilinear",
         "Spatial standardisation"),
        (normalised,  "gray",   "④  Normalise\nmean=0.5, std=0.5",
         "Range: [−1, 1] float32"),
    ]

    colors = [TEAL, ORG, MINT, PINK]
    for i, (img, cmap, title, sub) in enumerate(steps):
        ax = axes[i]
        if i < 3:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(img, cmap=cmap, vmin=-1, vmax=1)
        ax.axis("off")
        ax.set_title(title, color=colors[i], fontsize=11,
                     fontweight="bold", pad=6)
        ax.text(0.5, -0.08, sub, transform=ax.transAxes,
                ha="center", color=MUTED, fontsize=9)

        if i < 3:
            axes[i].annotate(
                "", xy=(1.08, 0.5), xytext=(1.0, 0.5),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=WHITE, lw=2)
            )

    plt.tight_layout(pad=2.0)
    plt.savefig(OUT / "fig5_preprocessing.png", dpi=130,
                bbox_inches="tight", facecolor=BG)
    plt.close()
    print("  saved fig5_preprocessing.png")

if __name__ == "__main__":
    try:
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    except ImportError:
        raise SystemExit("Run: pip install scikit-learn")

    print("Generating visuals...")
    fig_sample_slices()
    fig_reconstruction_error()
    fig_model_performance()
    fig_heatmap_localisation()
    fig_preprocessing_pipeline()
    print(f"\nAll done. Saved to: {OUT}/")
