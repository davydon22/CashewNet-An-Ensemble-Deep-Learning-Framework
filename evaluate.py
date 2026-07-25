"""
Ensemble + TTA inference, metrics, and statistical testing.

  - Adds Wilcoxon signed-rank test alongside the paired t-test —
    more defensible than a t-test alone when n_folds is small, since it
    doesn't assume the fold-level differences are normally distributed.
  - Adds bootstrap 95% CIs on accuracy/macro-F1 computed over individual test
    predictions (not fold means), giving a second, complementary uncertainty
    estimate that doesn't depend on having many folds.
  - `evaluate_test_set` now runs on the manifest's held-out `test` split only
    (never touched during k-fold training), reported once per experiment —
    not the repeated ad-hoc calls to compute_metrics() scattered.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, classification_report)

from datasets import ManifestDataset, MultiScaleTransform


def tta_transforms(x):
    return [x, torch.flip(x, dims=[3]), torch.flip(x, dims=[2]),
            torch.rot90(x, 1, [2, 3]), torch.rot90(x, 2, [2, 3]), torch.rot90(x, 3, [2, 3])]


class EnsembleModel:
    def __init__(self, models, device):
        self.models = [m.to(device).eval() for m in models]
        self.device = device

    @torch.no_grad()
    def predict_batch(self, images, use_tta=True):
        images = images.to(self.device)
        all_model_probs = []
        for model in self.models:
            views = tta_transforms(images) if use_tta else [images]
            view_probs = []
            for v in views:
                with autocast(device_type='cuda'):
                    probs = F.softmax(model(v), dim=1)
                view_probs.append(probs)
            all_model_probs.append(torch.stack(view_probs).mean(dim=0))
        return torch.stack(all_model_probs).mean(dim=0)


@torch.no_grad()
def tta_predict_single_model(model, manifest_df, device, cfg, use_tta=True):
    """Used by ablation.run_tta_only_ablation to isolate TTA from ensembling."""
    ds = ManifestDataset(manifest_df, MultiScaleTransform(cfg.img_size, is_train=False))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
    model = model.to(device).eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        views = tta_transforms(images) if use_tta else [images]
        probs = torch.stack([F.softmax(model(v), dim=1) for v in views]).mean(dim=0)
        all_preds.extend(probs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
    return 100 * accuracy_score(all_labels, all_preds)


def bootstrap_ci(labels, preds, metric_fn, n_boot=1000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    labels, preds = np.array(labels), np.array(preds)
    n = len(labels)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric_fn(labels[idx], preds[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(stats)), float(lo), float(hi)


def evaluate_test_set(ensemble, manifest_df, class_names, cfg, device):
    test_df = manifest_df[manifest_df.split == "test"]
    ds = ManifestDataset(test_df, MultiScaleTransform(cfg.img_size, is_train=False))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        probs = ensemble.predict_batch(images, use_tta=True)
        all_preds.extend(probs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    macro_prec = precision_score(all_labels, all_preds, average="macro")
    macro_rec = recall_score(all_labels, all_preds, average="macro")
    try:
        macro_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        macro_auc = float("nan")

    acc_mean, acc_lo, acc_hi = bootstrap_ci(all_labels, all_preds,
                                             lambda y, p: accuracy_score(y, p))
    f1_mean, f1_lo, f1_hi = bootstrap_ci(all_labels, all_preds,
                                          lambda y, p: f1_score(y, p, average="macro"))

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": acc, "macro_f1": macro_f1, "macro_precision": macro_prec,
        "macro_recall": macro_rec, "macro_auc": macro_auc,
        "accuracy_95ci": (acc_lo, acc_hi), "macro_f1_95ci": (f1_lo, f1_hi),
        "per_class_report": report, "confusion_matrix": cm,
        "labels": all_labels, "preds": all_preds, "probs": all_probs,
    }


def fold_level_ensemble_macro_f1(manifest_df, backbones, num_classes, cfg, device=None):
    """Build a genuine per-fold ensemble (that fold's own EfficientNetV2-S +
    ConvNeXt-Tiny + Swin-Tiny checkpoints) and evaluate it on that same
    fold's held-out val split. Returns a list of n_folds macro-F1 scores —
    directly comparable/pairable with each backbone's own per-fold scores
    from kfold_{backbone}.csv, since both are evaluated on identical val
    splits per fold.

    This exists because the ensemble itself was never run through k-fold CV
    (only individual backbones were) — but since every fold's checkpoint for
    every backbone is already saved to disk, no retraining is needed to
    reconstruct fold-level ensemble scores; this just loads and evaluates.
    """
    from models import build_model
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_scores = []
    for fold in sorted(manifest_df.fold.unique()):
        if fold < 0:  # test split is marked fold=-1 in the manifest, skip it
            continue
        fold_models = []
        for backbone in backbones:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"{backbone}_fold{fold}.pth")
            ckpt = torch.load(ckpt_path, map_location=device)
            m = build_model(backbone, num_classes)
            m.load_state_dict(ckpt["model_state_dict"])
            fold_models.append(m)
        ensemble = EnsembleModel(fold_models, device)

        val_df = manifest_df[(manifest_df.fold == fold) & (manifest_df.split == "val")]
        ds = ManifestDataset(val_df, MultiScaleTransform(cfg.img_size, is_train=False))
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

        all_preds, all_labels = [], []
        for images, labels in loader:
            probs = ensemble.predict_batch(images, use_tta=True)
            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
        fold_f1 = f1_score(all_labels, all_preds, average="macro")
        fold_scores.append(fold_f1)
        print(f"[fold_level_ensemble] fold {fold}: macro_f1={fold_f1:.4f}")
    return fold_scores


def significance_tests(cashewnet_fold_scores, baseline_fold_scores_dict):
    """
    paired t-test AND Wilcoxon signed-rank, computed over per-fold
    scores (e.g. macro-F1 per fold), for CashewNet vs. each baseline.

    cashewnet_fold_scores MUST be the ensemble's own per-fold scores (see
    fold_level_ensemble_macro_f1 above) — passing a single backbone's scores
    here instead produces a table that LOOKS like an ensemble comparison
    (rows are still labeled "CashewNet vs X")
    """
    rows = []
    for name, scores in baseline_fold_scores_dict.items():
        t_stat, t_p = ttest_rel(cashewnet_fold_scores, scores)
        try:
            w_stat, w_p = wilcoxon(cashewnet_fold_scores, scores)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        diff = np.array(cashewnet_fold_scores) - np.array(scores)
        cohens_d = diff.mean() / (diff.std(ddof=1) + 1e-12)
        rows.append({
            "comparison": f"CashewNet vs {name}",
            "t_stat": t_stat, "t_pvalue": t_p,
            "wilcoxon_stat": w_stat, "wilcoxon_pvalue": w_p,
            "cohens_d": cohens_d,
        })
    import pandas as pd
    return pd.DataFrame(rows)
