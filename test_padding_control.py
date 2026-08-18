"""
GradCAM++'s severe border-bias artifact to gradient backpropagation through the
backbone's "SAME"-style padding at strided convolutions. This script tests
that causal claim directly, rather than leavingit as an unverified attribution: 
it rebuilds the EfficientNetV2-S backbone
with all "SAME" padding replaced by "VALID" (no) padding, recomputes CAM
border-bias and localization metrics on the SAME set of annotated test
images, and reports the delta against the original SAME-padding numbers.

IMPORTANT CAVEAT: switching SAME->VALID changes the backbone's output
spatial dimensions at every strided layer, which cascades through the
network and means this VALID-padding model does NOT have the same trained
weights as the SAME-padding model evaluated elsewhere -- it
must be retrained (or at least fine-tuned) before its CAM outputs are
meaningful, since untrained/randomly-initialized weights would produce
meaningless attention maps unrelated to the padding hypothesis. This script
handles that: it fine-tunes the VALID-padding variant for a small number of
epochs on the same fold-0 training split used elsewhere in the ablation, 
not from scratch, keeping this a bounded "lightweight
control group" rather than a full second 5-fold training run.

Run from the project root, inside the container:
    python3 test_padding_control.py
"""
import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import timm

from config import CFG
from datasets import ManifestDataset, MultiScaleTransform
from models import build_model
from train import Trainer  # reuse the existing training loop, not a new one
from xai import compute_cam, get_last_conv_layer, heatmap_to_adaptive_mask, \
    pointing_game, border_bias_fraction, iou

OUT_DIR = CFG.output_dir
os.makedirs(OUT_DIR, exist_ok=True)


def convert_same_to_valid_padding(model):
    """Walks the backbone and replaces every Conv2dSame (timm's "SAME"
    padding implementation, which pads asymmetrically before a standard
    Conv2d to emulate TensorFlow's SAME behavior) with a plain Conv2d using
    VALID (zero) padding -- i.e. no explicit padding at all, matching the
    kernel/stride exactly with no compensating pad. This changes output
    spatial dimensions at every such layer, which is intentional: it is
    the whole point of the control."""
    import timm.layers as tlayers
    replaced = 0
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, tlayers.Conv2dSame) or type(child).__name__ == "Conv2dSame":
                new_conv = nn.Conv2d(
                    child.in_channels, child.out_channels, child.kernel_size,
                    stride=child.stride, padding=0, dilation=child.dilation,
                    groups=child.groups, bias=(child.bias is not None),
                )
                # Reuse the original weights so this is a padding-only change,
                # not a re-initialization -- keeps this a controlled comparison.
                with torch.no_grad():
                    new_conv.weight.copy_(child.weight)
                    if child.bias is not None:
                        new_conv.bias.copy_(child.bias)
                setattr(module, child_name, new_conv)
                replaced += 1
    print(f"Replaced {replaced} Conv2dSame layer(s) with VALID-padding Conv2d.")
    if replaced == 0:
        print("WARNING: no Conv2dSame layers found -- confirm this backbone actually "
              "uses SAME padding before trusting the rest of this script's output.")
    return model


def quick_finetune(model, train_df, device, epochs=3):
    """Short fine-tune (NOT full 50-epoch training) to let the model adapt
    to its now-different spatial dimensions after the padding swap. This is
    the 'lightweight' part of the control group -- long enough for the
    padding change's effect on attention patterns to be meaningfully
    evaluated, short enough to stay a bounded control rather than a second
    full experiment."""
    ds = ManifestDataset(train_df, MultiScaleTransform(CFG.img_size, is_train=True))
    loader = torch.utils.data.DataLoader(ds, batch_size=CFG.batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # low LR: adapting, not retraining from scratch
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  fine-tune epoch {epoch+1}/{epochs}: avg loss {total_loss/len(loader):.4f}")
    model.eval()
    return model


def evaluate_padding_variant(model, manifest_df, mask_dir, device, label="VALID-padding"):
    """Reuses the exact same evaluation logic as the main XAI evaluation
    (border_bias_fraction, adaptive IoU, pointing game) so results are
    directly comparable to Table 9's SAME-padding numbers."""
    rows = []
    for _, row in manifest_df.iterrows():
        cls = CFG.class_names[row["label"]]
        stem = os.path.splitext(os.path.basename(row["path"]))[0]
        mask_path = os.path.join(mask_dir, cls, f"{stem}.png")
        if not os.path.exists(mask_path):
            continue
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if raw_mask is None or (raw_mask > 0).mean() < 0.001:
            continue

        import torchvision.transforms.functional as TF
        mask_pil = Image.fromarray(raw_mask, mode="L")
        mask_pil = TF.resize(mask_pil, [256, 256], interpolation=TF.InterpolationMode.NEAREST)
        mask_pil = TF.center_crop(mask_pil, 224)
        gt_mask = np.array(mask_pil) > 0

        image = Image.open(row["path"]).convert("RGB")
        cam, pred_class = compute_cam(model, image, device, method="gradcam++")
        cam_resized = cv2.resize(cam, (224, 224))

        pred_mask_adaptive = heatmap_to_adaptive_mask(cam_resized)
        iou_adaptive = iou(pred_mask_adaptive, gt_mask)
        hit = pointing_game(cam_resized, gt_mask)
        border_bias = border_bias_fraction(cam_resized)

        rows.append({"class": cls, "iou_adaptive": iou_adaptive,
                     "pointing_game_hit": hit, "border_bias_fraction": border_bias})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No matching masks found -- check mask_dir and manifest paths.")
    summary = df.groupby("class").agg(
        iou_adaptive_mean=("iou_adaptive", "mean"),
        pointing_game_hit_rate=("pointing_game_hit", "mean"),
        border_bias_mean=("border_bias_fraction", "mean"),
        n=("iou_adaptive", "count"),
    ).reset_index()
    summary["variant"] = label
    return df, summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(f"{CFG.manifest_dir}/manifest.csv")
    # BUGFIX: the manifest is long-format -- every train/val image appears once
    # per fold, with 'split' indicating train/val WITHIN that fold's own context
    # (confirmed via diagnose_fold_column.py). Fold 0's own training set is
    # therefore (fold == 0) & (split == 'train'), NOT (split == 'train') &
    # (fold != 0) as an earlier version of this script had it -- that filter
    # pulled a meaningless mix of rows from multiple different folds' training
    # contexts rather than fold 0's actual training set.
    fold0_train = manifest[(manifest.fold == 0) & (manifest.split == "train")]
    test_df = manifest[manifest.split == "test"]

    print("Building EfficientNetV2-S with VALID padding (weights copied from SAME-padding init)...")
    model = build_model("tf_efficientnetv2_s", num_classes=len(CFG.class_names), pretrained=True).to(device)
    model = convert_same_to_valid_padding(model)
    model = model.to(device)  # BUGFIX: convert_same_to_valid_padding creates new nn.Conv2d
    # layers via setattr, which default to CPU regardless of the rest of the model's
    # device -- without this second .to(device) call, the replaced layers stay on CPU
    # while everything else (and the input batch) is on CUDA, causing exactly the
    # "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)" crash.

    print("Quick fine-tune (3 epochs, low LR) to adapt to new spatial dimensions...")
    model = quick_finetune(model, fold0_train, device, epochs=3)

    print("Evaluating VALID-padding variant's CAM localization/border-bias...")
    raw_df, summary_valid = evaluate_padding_variant(
        model, test_df, mask_dir="lesion_masks", device=device, label="VALID-padding"
    )
    summary_valid.to_csv(f"{OUT_DIR}/padding_control_valid.csv", index=False)
    print(summary_valid.to_string(index=False))
    print(f"\nSaved {OUT_DIR}/padding_control_valid.csv")
    print("\nCompare this directly against the existing SAME-padding GradCAM++ rows in "
          "cashewnet_outputs/xai_iou_summary.csv (border_bias_mean column) -- if the "
          "border-bias hypothesis is correct, border_bias_mean here should be "
          "substantially lower than the 0.54-0.62 range reported for SAME padding.")


if __name__ == "__main__":
    main()
