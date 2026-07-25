"""
Real ablation study.

This module:
  1. Runs the component ablation (baseline -> +ECA -> +fusion -> +mixup ->
     +EMA -> +focal_loss -> ensemble) on ALL THREE backbones, not just one,
     so "does ECA help" can be answered per-architecture, not assumed to
     generalize from a single case.
  2. Adds a "TTA only, no ensemble" row per backbone, isolating what TTA buys
     you from what multi-model ensembling buys you.
  3. Writes one flat CSV others can pivot/plot however they like, instead of
     hand-typed dictionaries that can drift from what was actually run.
"""
import itertools
import os
import pandas as pd
import torch

from config import CFG
from train import run_kfold, train_one_config, resume_from_checkpoint
from evaluate import tta_predict_single_model


ABLATION_STEPS = [
    # (label, use_eca, use_fusion, use_mixup, use_ema, loss_type)
    ("baseline",              False, False, False, False, "cross_entropy"),
    ("+ECA",                  True,  False, False, False, "cross_entropy"),
    ("+ECA+Fusion",           True,  True,  False, False, "cross_entropy"),
    ("+ECA+Fusion+Mixup",     True,  True,  True,  False, "cross_entropy"),
    ("+ECA+Fusion+Mixup+EMA", True,  True,  True,  True,  "cross_entropy"),
    ("Full (+FocalLoss+LS)",  True,  True,  True,  True,  "focal_ls"),
]


def run_component_ablation(manifest_df, num_classes, cfg=CFG, backbones=None, resume=True):
    """Runs every step in ABLATION_STEPS for every backbone. Uses fold 0 only
    for the ablation sweep (component ablation is about relative deltas, not
    final reported accuracy — the *main* result still uses the full 5-fold
    ensemble). This keeps compute bounded: 3 backbones x 6 steps x 1 fold,
    instead of x5 folds.

    resume=True (default): each step's checkpoint is tagged
    ablation_{backbone}_{step}_fold0.pth. If that file already exists on
    disk — e.g. because a previous run of this sweep was interrupted midway
    through — that step is not retrained; its result is reconstructed from
    the saved weights instead, the same way run_kfold() resumes."""
    backbones = backbones or cfg.backbones
    rows = []
    for backbone in backbones:
        for label, use_eca, use_fusion, use_mixup, use_ema, loss_type in ABLATION_STEPS:
            tag = f"ablation_{backbone}_{label}"
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"{tag}_fold0.pth")

            if resume and os.path.exists(ckpt_path):
                res = resume_from_checkpoint(manifest_df, fold=0, backbone=backbone,
                                              num_classes=num_classes, ckpt_path=ckpt_path,
                                              cfg=cfg, use_eca=use_eca, use_fusion=use_fusion)
            else:
                # Train fold 0 only, directly — run_kfold would train all
                # cfg.n_folds (5) folds per step, 5x more compute than needed.
                res = train_one_config(
                    manifest_df, fold=0, backbone=backbone, num_classes=num_classes, cfg=cfg,
                    use_eca=use_eca, use_fusion=use_fusion, use_mixup=use_mixup,
                    use_ema=use_ema, loss_type=loss_type, tag=tag,
                )
            fv = res["final_val"]
            rows.append({
                "backbone": backbone, "step": label,
                "accuracy": fv["accuracy"], "macro_f1": fv["macro_f1"],
                "macro_auc": fv["macro_auc"],
            })
    return pd.DataFrame(rows)


def run_tta_only_ablation(manifest_df, num_classes, fold_models, best_fold_idx, cfg=CFG, device=None):
    """Isolate TTA's contribution from ensembling's contribution:
    for each single trained backbone, report accuracy with vs without TTA,
    using the SAME held-out fold predictions the main ensemble draws from.

    best_fold_idx: {backbone: fold} — REQUIRED, not optional. Each backbone's
    best model was selected from a specific fold (whichever had the highest
    macro_f1 during k-fold CV), and must be evaluated against THAT fold's val
    split, not a fixed fold. Evaluating against the wrong fold's val split
    means testing partly on images that were in that model's training set
    for every fold other than its own — this produced inflated ~99.9%
    "no_tta" numbers in an earlier run of this function, versus the ~98%
    the component ablation table shows for the same backbones/checkpoints."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for backbone, model in fold_models.items():
        fold = best_fold_idx[backbone]
        val_df = manifest_df[(manifest_df.fold == fold) & (manifest_df.split == "val")]
        no_tta_acc, tta_acc = tta_predict_single_model(model, val_df, device, cfg, use_tta=False), \
                               tta_predict_single_model(model, val_df, device, cfg, use_tta=True)
        rows.append({"backbone": backbone, "fold_used": fold,
                     "accuracy_no_tta": no_tta_acc, "accuracy_with_tta": tta_acc,
                     "tta_delta": tta_acc - no_tta_acc})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    manifest = pd.read_csv(f"{CFG.manifest_dir}/manifest.csv")
    ablation_df = run_component_ablation(manifest, len(CFG.class_names))
    ablation_df.to_csv(f"{CFG.output_dir}/component_ablation_all_backbones.csv", index=False)
    print(ablation_df.pivot(index="step", columns="backbone", values="accuracy"))
