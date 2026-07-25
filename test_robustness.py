"""
Evaluates the trained ensemble's robustness to synthetic perturbations of
the held-out test set: brightness/contrast shifts (illumination), Gaussian
blur (motion/focus blur proxy), random rectangular occlusion patches, and
Gaussian pixel noise (low-quality-image proxy). 

This does NOT include real early-stage-symptom evaluation (that requires
actual early-stage disease photos, which is a data-collection task, not a
perturbation you can synthesize) — see the note in main() for how to add
that once such images are available.

Run from the project root, inside the container:
    python3 test_robustness.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CFG
from datasets import MultiScaleTransform
from evaluate import EnsembleModel

OUT_DIR = CFG.output_dir
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Perturbation functions — each takes a PIL image, returns a PIL image
# ============================================================

def perturb_brightness_up(img, severity):
    factor = 1.0 + severity * 0.4  # up to +40% at severity=1.0
    return ImageEnhance.Brightness(img).enhance(factor)

def perturb_brightness_down(img, severity):
    factor = 1.0 - severity * 0.4  # down to -40% at severity=1.0
    return ImageEnhance.Brightness(img).enhance(factor)

def perturb_contrast(img, severity):
    factor = 1.0 - severity * 0.5
    return ImageEnhance.Contrast(img).enhance(factor)

def perturb_blur(img, severity):
    radius = severity * 4.0  # up to radius=4 Gaussian blur
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def perturb_occlusion(img, severity, seed=None):
    rng = np.random.RandomState(seed)
    img = img.copy()
    w, h = img.size
    patch_frac = 0.1 + severity * 0.3  # occlude 10%-40% of the frame
    pw, ph = int(w * patch_frac), int(h * patch_frac)
    x0 = rng.randint(0, max(1, w - pw))
    y0 = rng.randint(0, max(1, h - ph))
    arr = np.array(img)
    arr[y0:y0 + ph, x0:x0 + pw] = rng.randint(0, 255, size=(ph, pw, 3), dtype=np.uint8) if arr.ndim == 3 else 128
    return Image.fromarray(arr)

def perturb_noise(img, severity, seed=None):
    rng = np.random.RandomState(seed)
    arr = np.array(img).astype(np.float32)
    noise_std = severity * 40.0  # up to std=40 (out of 255) at severity=1.0
    noisy = arr + rng.normal(0, noise_std, arr.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))


PERTURBATIONS = {
    "brightness_up": perturb_brightness_up,
    "brightness_down": perturb_brightness_down,
    "contrast_down": perturb_contrast,
    "blur": perturb_blur,
    "occlusion": perturb_occlusion,
    "noise": perturb_noise,
}
SEVERITIES = [0.0, 0.33, 0.66, 1.0]  # 0.0 = clean baseline, no perturbation


class PerturbedTestDataset(Dataset):
    """Applies a named perturbation at a given severity to each image
    BEFORE the standard eval-time resize/crop transform, so the perturbation
    happens at native resolution (more realistic) rather than on an
    already-downsampled 224x224 image."""
    def __init__(self, df, perturb_fn, severity, transform, seed_offset=0):
        self.df = df.reset_index(drop=True)
        self.perturb_fn = perturb_fn
        self.severity = severity
        self.transform = transform
        self.seed_offset = seed_offset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.severity > 0:
            try:
                img = self.perturb_fn(img, self.severity, seed=idx + self.seed_offset)
            except TypeError:
                img = self.perturb_fn(img, self.severity)
        img = self.transform(img)
        return img, int(row["label"])


def evaluate_perturbed(ensemble, df, perturb_name, severity, device, seed_offset=0):
    transform = MultiScaleTransform(CFG.img_size, is_train=False)
    perturb_fn = PERTURBATIONS[perturb_name] if severity > 0 else (lambda img, s, seed=None: img)
    ds = PerturbedTestDataset(df, perturb_fn, severity, transform, seed_offset=seed_offset)
    loader = DataLoader(ds, batch_size=CFG.batch_size, shuffle=False, num_workers=4)

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            probs = ensemble.predict_batch(images, use_tta=True)
            preds = probs.argmax(dim=1).cpu()
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(f"{CFG.manifest_dir}/manifest.csv")
    test_df = manifest[manifest.split == "test"].reset_index(drop=True)

    best_models = torch.load(f"{CFG.checkpoint_dir}/best_models.pt", weights_only=False)
    ensemble = EnsembleModel(list(best_models.values()), device)

    print("Evaluating clean (unperturbed) baseline...")
    clean_acc = evaluate_perturbed(ensemble, test_df, None, 0.0, device)
    print(f"  Clean test accuracy: {clean_acc*100:.2f}%")

    rows = [{"perturbation": "clean", "severity": 0.0, "accuracy": clean_acc}]
    for name in PERTURBATIONS:
        for severity in SEVERITIES[1:]:
            print(f"Evaluating {name} at severity {severity}...")
            acc = evaluate_perturbed(ensemble, test_df, name, severity, device)
            print(f"  accuracy: {acc*100:.2f}% (delta: {(acc-clean_acc)*100:+.2f} pp)")
            rows.append({"perturbation": name, "severity": severity, "accuracy": acc})

    df = pd.DataFrame(rows)
    df["accuracy_pct"] = df["accuracy"] * 100
    df["delta_pp"] = (df["accuracy"] - clean_acc) * 100
    df.to_csv(f"{OUT_DIR}/robustness_results.csv", index=False)
    print(f"\nSaved {OUT_DIR}/robustness_results.csv")

    # Plot: accuracy vs severity, one line per perturbation type
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in PERTURBATIONS:
        sub = df[df.perturbation == name]
        xs = [0.0] + sub["severity"].tolist()
        ys = [clean_acc * 100] + sub["accuracy_pct"].tolist()
        ax.plot(xs, ys, marker="o", label=name)
    ax.axhline(clean_acc * 100, color="black", linestyle="--", linewidth=1, label="clean baseline")
    ax.set_xlabel("Perturbation severity")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Ensemble robustness to synthetic image perturbations")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/robustness_curves.png", dpi=200, facecolor="white")
    print(f"Saved {OUT_DIR}/robustness_curves.png")

    print("\nNOTE: this covers illumination/contrast, blur, occlusion, and noise robustness. "
          "It does NOT cover early-stage-symptom robustness specifically, "
          "since that requires real early-stage disease photos rather than a synthetic "
          "perturbation — if you have or can collect such images, place them in a separate "
          "held-out folder and evaluate the ensemble on them directly (same pattern as "
          "generate_test_predictions_and_figures.py), rather than trying to simulate early-stage "
          "symptoms synthetically.")


if __name__ == "__main__":
    main()
