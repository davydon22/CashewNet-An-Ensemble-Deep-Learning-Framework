"""
Runs all three CAM methods on the SAME image and prints/visualizes them
side by side, to determine whether the row=0 peak artifact is specific to
one method (e.g. gradient-based GradCAM++) or affects all of them equally
— which would point toward an architecture-level cause (tf_efficientnetv2_s
uses TensorFlow-style "SAME" padding at its strided downsampling layers,
a known source of border-bias in gradient-based CAM methods for TF-ported
architectures) rather than a target-layer selection bug.

Run from the project root, inside the container:
    python3 debug_single_xai_allmethods.py
"""
import pandas as pd
import numpy as np
import cv2
import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CFG
from xai import compute_cam, get_last_conv_layer

MIN_LESION_FRACTION = 0.001
METHODS = ["gradcam++", "scorecam", "eigencam"]


def main():
    sel = pd.read_csv("annotation_selection_manifest.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_models = torch.load(f"{CFG.checkpoint_dir}/best_models.pt", weights_only=False)
    model = best_models[CFG.backbones[0]].to(device).eval()

    chosen = None
    for _, row in sel.iterrows():
        mask_path = row["mask_should_go_to"]
        if not os.path.exists(mask_path):
            continue
        raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if raw is not None and (raw > 127).mean() >= MIN_LESION_FRACTION:
            chosen = row
            break

    if chosen is None:
        print("No real annotated masks found.")
        return

    print(f"Debugging: {chosen['staged_path']} (class={chosen['class']})")
    mask_path = chosen["mask_should_go_to"]
    source_path = chosen["source_path"]

    raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_pil = Image.fromarray(raw_mask, mode="L")
    mask_pil = TF.resize(mask_pil, [256, 256], interpolation=TF.InterpolationMode.NEAREST)
    mask_pil = TF.center_crop(mask_pil, 224)
    gt_mask = np.array(mask_pil) > 127
    ys, xs = np.where(gt_mask)
    print(f"gt_mask bounding box: rows [{ys.min()}-{ys.max()}], cols [{xs.min()}-{xs.max()}]")

    image = Image.open(source_path).convert("RGB")

    fig, axes = plt.subplots(1, len(METHODS) + 1, figsize=(5 * (len(METHODS) + 1), 5))
    axes[0].imshow(gt_mask, cmap="gray")
    axes[0].set_title("Ground truth mask")

    for i, method in enumerate(METHODS, start=1):
        cam, pred_class = compute_cam(model, image, device, method=method)
        peak_idx = np.unravel_index(np.argmax(cam), cam.shape)
        hit = gt_mask[peak_idx]
        print(f"\n{method}: peak at (row={peak_idx[0]}, col={peak_idx[1]}), "
              f"value={cam[peak_idx]:.4f}, hit={bool(hit)}")
        print(f"  activation stats: min={cam.min():.4f} max={cam.max():.4f} "
              f"mean={cam.mean():.4f} std={cam.std():.4f}")
        # How much of the activation mass sits in the outermost 10% border
        # vs the interior — a high border fraction would support the
        # "SAME"-padding boundary-artifact theory.
        border = max(1, int(0.1 * cam.shape[0]))
        border_mask = np.zeros_like(cam, dtype=bool)
        border_mask[:border, :] = border_mask[-border:, :] = True
        border_mask[:, :border] = border_mask[:, -border:] = True
        border_fraction = cam[border_mask].sum() / max(cam.sum(), 1e-8)
        print(f"  fraction of total activation mass in outer 10% border: {border_fraction:.3f} "
              f"(border is only ~{border_mask.mean()*100:.0f}% of pixels, so >{border_mask.mean():.2f} indicates bias)")

        axes[i].imshow(cam, cmap="jet")
        axes[i].scatter([peak_idx[1]], [peak_idx[0]], c="white", marker="x", s=200, linewidths=3)
        axes[i].set_title(f"{method}\npeak=({peak_idx[0]},{peak_idx[1]}) hit={bool(hit)}")

    plt.tight_layout()
    out_path = f"{CFG.output_dir}/xai_debug_allmethods_overlay.png"
    plt.savefig(out_path, dpi=100)
    print(f"\n✅ Saved side-by-side comparison to {out_path}")


if __name__ == "__main__":
    main()
