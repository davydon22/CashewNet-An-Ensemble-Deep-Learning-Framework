"""
Quantitative explainability

  1. Runs GradCAM++, Score-CAM, and Eigen-CAM on the same images.
  2. Thresholds each heatmap to a binary mask and computes IoU against a
     human-annotated ground-truth lesion mask — this requires the
     ~120-150 manually annotated lesion masks; point cfg.mask_dir at that folder once it exists.
  3. Reports mean +/- std IoU **per class**.

Expected mask folder layout (binary PNG, 255=lesion, 0=background, same
filename stem as the source image):
    lesion_masks/<class_name>/<image_stem>.png
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import cv2
import pandas as pd

from pytorch_grad_cam import GradCAMPlusPlus, ScoreCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from datasets import MultiScaleTransform

CAM_METHODS = {"gradcam++": GradCAMPlusPlus, "scorecam": ScoreCAM, "eigencam": EigenCAM}


def get_last_conv_layer(model):
    """Returns the module that produces the backbone's actual final spatial
    feature map — the correct GradCAM target layer.
    ."""
    if hasattr(model.backbone, "feature_info"):
        last_module_name = model.backbone.feature_info.info[-1]["module"]
        try:
            return model.backbone.get_submodule(last_module_name)
        except AttributeError:
            pass  # fall through to the Conv2d search below as a last resort

    for layer in reversed(list(model.backbone.modules())):
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    raise ValueError("No Conv2d layer found in backbone — for pure-transformer "
                      "backbones (Swin), use the reshape_transform hook per "
                      "pytorch-grad-cam's ViT/Swin documentation instead.")


def compute_cam(model, image_pil, device, method="gradcam++", target_layer=None):
    transform = MultiScaleTransform(224, is_train=False)
    input_tensor = transform(image_pil).unsqueeze(0).to(device)
    model = model.to(device).eval()
    target_layer = target_layer or get_last_conv_layer(model)

    with torch.no_grad():
        pred_class = F.softmax(model(input_tensor), dim=1).argmax(dim=1).item()

    cam_cls = CAM_METHODS[method]
    cam = cam_cls(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
    return grayscale_cam, pred_class


def heatmap_to_binary_mask(grayscale_cam, top_fraction=0.2):
    """Threshold at the value that keeps the top `top_fraction` of activation
    mass — more robust than a fixed absolute threshold across images with
    different overall activation scales.

    CAVEAT: this forces every image's predicted region to the same fixed
    area (top_fraction of all pixels), regardless of how large the actual
    lesion is. For a class like leaf_miner, whose real lesions cover only
    ~1% of the image (thin trails), a fixed 20% predicted region caps the
    achievable IoU near ~0.06 even for a hypothetically perfect model — so
    IoU alone cannot distinguish "the model looks in the wrong place" from
    "the predicted region is simply much larger than the true lesion by
    construction." See heatmap_to_adaptive_mask and pointing_game below for
    two area-independent complements that don't share this limitation."""
    flat = grayscale_cam.flatten()
    k = max(1, int(len(flat) * top_fraction))
    thresh = np.partition(flat, -k)[-k]
    return (grayscale_cam >= thresh).astype(np.uint8)


def heatmap_to_adaptive_mask(grayscale_cam):
    """Otsu's method finds a natural break point in the CAM's own activation
    histogram, so the predicted region's size reflects that specific image's
    activation distribution rather than an arbitrary fixed fraction — a
    heatmap with one small, sharply-peaked hot spot gets a small predicted
    region; one with broad, diffuse activation gets a larger one. This is
    the standard alternative to fixed-top-fraction thresholding in the CAM
    evaluation literature specifically because of the area-mismatch issue
    noted above."""
    normalized = cv2.normalize(grayscale_cam, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (mask > 0).astype(np.uint8)


def pointing_game(grayscale_cam, gt_mask):
    """ "pointing game": does the CAM's single highest-
    activation pixel fall inside the ground-truth lesion region? Binary
    hit (1) or miss (0) per image — completely insensitive to relative
    area, since it only checks a single point, not a region overlap. This
    is the standard complement to IoU-style metrics precisely because it
    cannot be confounded by lesion-size vs. predicted-region-size mismatch
    the way IoU can."""
    peak_idx = np.unravel_index(np.argmax(grayscale_cam), grayscale_cam.shape)
    return int(gt_mask[peak_idx] > 0)


def iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return inter / union if union > 0 else float("nan")


def border_bias_fraction(grayscale_cam, border_frac=0.1):
    """Fraction of a CAM's total activation mass that sits within its outer
    `border_frac` margin, versus that margin's actual share of pixels
    (e.g. ~0.35 for a 10% border on a square image). A value well above the
    border's pixel share indicates the CAM is disproportionately drawn to
    image edges — a known failure mode for gradient-based CAM methods on
    architectures using "SAME"-style padding at strided convolutions (like
    tf_efficientnetv2_s), where gradients backpropagated through padded
    layers can create spurious edge activation unrelated to genuine image
    content. Confirmed via debug_single_xai_allmethods.py: GradCAM++ showed
    0.675 mass in a 0.35-share border region on this backbone (severe
    bias), while Score-CAM (0.353, matches uniform) and EigenCAM (0.034,
    interior-concentrated) did not — because only GradCAM++ backpropagates
    gradients through the padded layers; Score-CAM (perturbation-based) and
    EigenCAM (activation-PCA-based) don't rely on gradients at all."""
    h, w = grayscale_cam.shape
    border = max(1, int(border_frac * h))
    border_mask = np.zeros_like(grayscale_cam, dtype=bool)
    border_mask[:border, :] = border_mask[-border:, :] = True
    border_mask[:, :border] = border_mask[:, -border:] = True
    total = grayscale_cam.sum()
    return grayscale_cam[border_mask].sum() / total if total > 0 else float("nan")


def evaluate_xai_iou(model, manifest_df, class_names, mask_dir, device,
                      methods=("gradcam++", "scorecam", "eigencam"), top_fraction=0.2,
                      min_lesion_fraction=0.001):
    """
    manifest_df must contain a subset of images that also have a corresponding
    file in mask_dir/<class>/<stem>.png. Rows without a mask are skipped.
    """
    rows = []
    skipped_empty = 0
    skipped_healthy = 0
    for _, row in manifest_df.iterrows():
        cls = class_names[row["label"]]
        stem = os.path.splitext(os.path.basename(row["path"]))[0]
        mask_path = os.path.join(mask_dir, cls, f"{stem}.png")
        if not os.path.exists(mask_path):
            continue

        if cls == "healthy":
            # Lesion-localization IoU/pointing-game is structurally
            # meaningless here: a genuinely healthy leaf has no lesion, so
            # the ground truth is correctly an empty mask, not an
            # unannotated placeholder. IoU against an empty set is always
            # 0, and pointing-game is always a miss — not because the
            # model's attention is wrong, but because there is nothing to
            # overlap with. Including these would either (a) get wrongly
            # skipped as if unannotated, discarding real completed work, or
            # (b) get included and mechanically deflate every method's
            # aggregate scores for a reason unrelated to genuine
            # localization quality. Excluded from this specific evaluation;
            # a diffuseness/entropy-based metric would be the appropriate
            # (separate) way to quantify whether the model avoids false
            # localization on healthy leaves, not IoU/pointing-game.
            skipped_healthy += 1
            continue

        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if raw_mask is None or (raw_mask > 0).mean() < min_lesion_fraction:
            skipped_empty += 1
            continue
        # Must replicate MultiScaleTransform's eval-time geometry EXACTLY
        # (resize to 256 -> center-crop to 224), not a naive direct resize
        # to 224 — the model's input, and therefore the CAM's spatial
        # output, only ever "sees" the center-cropped region. A mask
        # resized straight from the full original frame to 224x224 instead
        # represents a DIFFERENT crop of the same photo, so comparing the
        # two directly compares misaligned coordinate spaces, not the same
        # region twice. NEAREST interpolation avoids blurring the binary
        # mask's edges into ambiguous gray values during the resize.
        mask_pil = Image.fromarray(raw_mask, mode="L")
        mask_pil = TF.resize(mask_pil, [256, 256], interpolation=TF.InterpolationMode.NEAREST)
        mask_pil = TF.center_crop(mask_pil, 224)
        # >0, not >127: this annotation export uses small nonzero integer
        # values (e.g. 14/38/75/113) as class-index markers rather than
        # near-white (255) intensity — confirmed by inspecting real
        # exported masks, where files with genuine, spatially-coherent
        # lesion annotations (verified: correct bounding box, plausible
        # ~1-3% lesion-area fraction matching known leaf_miner scale) had
        # a maximum pixel value well under 127. A genuinely empty mask has
        # EVERY pixel at exactly 0 regardless of which convention is used
        # for "marked," so >0 is the correct, convention-agnostic check.
        gt_mask = np.array(mask_pil) > 0

        image = Image.open(row["path"]).convert("RGB")
        for method in methods:
            cam, pred_class = compute_cam(model, image, device, method=method)
            cam_resized = cv2.resize(cam, (224, 224))

            pred_mask_fixed = heatmap_to_binary_mask(cam_resized, top_fraction)
            iou_fixed = iou(pred_mask_fixed, gt_mask)

            pred_mask_adaptive = heatmap_to_adaptive_mask(cam_resized)
            iou_adaptive = iou(pred_mask_adaptive, gt_mask)

            hit = pointing_game(cam_resized, gt_mask)
            border_bias = border_bias_fraction(cam_resized)

            rows.append({"path": row["path"], "class": cls, "method": method,
                         "iou_fixed_top20pct": iou_fixed, "iou_adaptive_otsu": iou_adaptive,
                         "pointing_game_hit": hit, "border_bias_fraction": border_bias,
                         "predicted_correctly": pred_class == row["label"]})

    if skipped_empty > 0:
        print(f"⚠️ Skipped {skipped_empty} mask file(s) that exist on disk but "
              f"are empty/below {min_lesion_fraction*100:.1f}% lesion coverage — "
              f"these are unannotated placeholders, not real 'no lesion' "
              f"ground truth, and would have deflated the IoU averages if "
              f"included. Run check_mask_progress.py to see which specific "
              f"files these are.")
    if skipped_healthy > 0:
        print(f"ℹ️  Excluded {skipped_healthy} healthy-class image(s) from "
              f"lesion-localization IoU/pointing-game — there's no lesion to "
              f"localize on a healthy leaf, so this metric doesn't apply "
              f"(this is expected, not a data problem).")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"No valid (non-empty) matching masks found under {mask_dir}. "
            f"Populate {mask_dir}/<class>/<image_stem>.png with REAL lesion "
            f"annotations (not empty placeholders) for at least a subset of "
            f"test images before running this (this is the ~120-150 image "
            f"lesion annotation task)."
        )
    summary = df.groupby(["class", "method"]).agg(
        iou_fixed_mean=("iou_fixed_top20pct", "mean"),
        iou_fixed_std=("iou_fixed_top20pct", "std"),
        iou_adaptive_mean=("iou_adaptive_otsu", "mean"),
        iou_adaptive_std=("iou_adaptive_otsu", "std"),
        pointing_game_hit_rate=("pointing_game_hit", "mean"),
        border_bias_mean=("border_bias_fraction", "mean"),
        count=("iou_fixed_top20pct", "count"),
    ).reset_index()
    return df, summary
