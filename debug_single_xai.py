"""
Maximally verbose single-image debug for the XAI pipeline. Picks ONE real
annotated mask, runs the exact same compute_cam() the real pipeline uses,
and prints/saves everything needed to see AT A GLANCE whether the CAM's
peak activation and the ground-truth lesion are even landing in the same
rough part of the image — rather than inferring this indirectly from
aggregate metrics.

Run from the project root, inside the container (needs the real model +
GPU):
    python3 debug_single_xai.py
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


def main():
    sel = pd.read_csv("annotation_selection_manifest.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_models = torch.load(f"{CFG.checkpoint_dir}/best_models.pt", weights_only=False)
    model = best_models[CFG.backbones[0]].to(device).eval()
    print(f"Using backbone: {CFG.backbones[0]}")
    print(f"Target layer: {get_last_conv_layer(model)}")

    # Find the first real (non-placeholder) mask to debug
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
        print("No real annotated masks found — annotate at least one first.")
        return

    print(f"\nDebugging: {chosen['staged_path']} (class={chosen['class']})")
    mask_path = chosen["mask_should_go_to"]
    source_path = chosen["source_path"]

    raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    print(f"Raw mask shape: {raw_mask.shape}, lesion pixels: {(raw_mask > 127).sum()}")

    mask_pil = Image.fromarray(raw_mask, mode="L")
    mask_pil = TF.resize(mask_pil, [256, 256], interpolation=TF.InterpolationMode.NEAREST)
    mask_pil = TF.center_crop(mask_pil, 224)
    gt_mask = np.array(mask_pil) > 127
    print(f"Cropped gt_mask shape: {gt_mask.shape}, dtype: {gt_mask.dtype}")
    ys, xs = np.where(gt_mask)
    if len(ys) > 0:
        print(f"gt_mask bounding box: rows [{ys.min()}-{ys.max()}], cols [{xs.min()}-{xs.max()}]")
        print(f"gt_mask centroid: (row={ys.mean():.1f}, col={xs.mean():.1f})")
    else:
        print("gt_mask is EMPTY after crop (lesion fully cropped away for this image)")

    image = Image.open(source_path).convert("RGB")
    print(f"\nSource image size: {image.size}")

    grayscale_cam, pred_class = compute_cam(model, image, device, method="gradcam++")
    print(f"\nRaw CAM shape: {grayscale_cam.shape}, dtype: {grayscale_cam.dtype}")
    print(f"CAM min/max: {grayscale_cam.min():.4f} / {grayscale_cam.max():.4f}")
    print(f"Predicted class index: {pred_class} ({CFG.class_names[pred_class]})")
    print(f"True class index: {chosen['class']} -> "
          f"{'MATCH' if CFG.class_names[pred_class] == chosen['class'] else 'MISMATCH'}")

    peak_idx = np.unravel_index(np.argmax(grayscale_cam), grayscale_cam.shape)
    print(f"\nCAM peak activation location: (row={peak_idx[0]}, col={peak_idx[1]}), "
          f"value={grayscale_cam[peak_idx]:.4f}")
    print(f"Is peak inside gt_mask (pointing game)? {gt_mask[peak_idx] if gt_mask.shape == grayscale_cam.shape else 'SHAPE MISMATCH'}")

    if gt_mask.shape != grayscale_cam.shape:
        print(f"\n🚨 SHAPE MISMATCH: gt_mask {gt_mask.shape} vs grayscale_cam "
              f"{grayscale_cam.shape} — THIS IS THE BUG if shapes differ.")

    # Save a visual overlay — fastest way to spot a transpose/flip/offset bug
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gt_mask, cmap="gray")
    axes[0].set_title(f"Ground truth mask\nbbox center: row={ys.mean():.0f}, col={xs.mean():.0f}" if len(ys)>0 else "EMPTY")
    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].scatter([peak_idx[1]], [peak_idx[0]], c="white", marker="x", s=200, linewidths=3)
    axes[1].set_title(f"GradCAM++ (white X = peak)\npeak: row={peak_idx[0]}, col={peak_idx[1]}")
    axes[2].imshow(gt_mask, cmap="gray", alpha=0.5)
    axes[2].imshow(grayscale_cam, cmap="jet", alpha=0.5)
    axes[2].scatter([peak_idx[1]], [peak_idx[0]], c="white", marker="x", s=200, linewidths=3)
    axes[2].set_title("Overlay (mask + CAM + peak)")
    plt.tight_layout()
    out_path = f"{CFG.output_dir}/xai_debug_overlay.png"
    plt.savefig(out_path, dpi=100)
    print(f"\n✅ Saved visual overlay to {out_path} — pull this file and look at it directly, "
          f"this will make any transpose/flip/offset bug immediately obvious.")


if __name__ == "__main__":
    main()
