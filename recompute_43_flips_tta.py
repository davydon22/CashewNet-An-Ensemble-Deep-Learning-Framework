"""
Recomputes fold-level ensemble significance test under
flips-only TTA. Confirmed via check_fold_level_tta_usage.py that the
existing fold_level_ensemble_macro_f1 function uses
ensemble.predict_batch(images, use_tta=True) -- the old 6-view scheme --
so this section needs recomputing for consistency with the new headline
result.

This reuses the exact per-fold checkpoint-loading structure of the real
fold_level_ensemble_macro_f1 (same fold loop, same checkpoint paths, same
val-split filtering) but replaces the use_tta=True call with the verified
flips-only view-averaging logic used in the other new scripts, so the
fold-level construction itself is unchanged -- only the TTA scheme inside
it differs.

Individual backbones' own per-fold scores are NOT recomputed
here: they come from raw validation during k-fold training, which does not
use TTA at all (confirmed in the earlier audit) and are therefore
unaffected by this change. Their already-established values are hardcoded
below for the paired significance test.

Run from the project root, inside the container:
    python3 recompute_section43_flips_tta.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from scipy.stats import ttest_rel, wilcoxon

from config import CFG
from datasets import ManifestDataset, MultiScaleTransform
from models import build_model
from evaluate import EnsembleModel

OUT_DIR = CFG.output_dir

FLIPS_ONLY_VIEWS = [
    ("identity", None),
    ("hflip", Image.FLIP_LEFT_RIGHT),
    ("vflip", Image.FLIP_TOP_BOTTOM),
]

# Already-established per-fold macro-F1 scores for each backbone
# earlier kfold_{backbone}.csv results) -- these do NOT use TTA (raw k-fold
# validation) and are unaffected by the TTA scheme change, so they are not
# recomputed here.
BACKBONE_FOLD_SCORES = {
    "tf_efficientnetv2_s": [0.983980, 0.990549, 0.989164, 0.986525, 0.988832],
    "convnext_tiny": [0.983151, 0.979595, 0.988039, 0.983234, 0.952264],
    "swin_tiny_patch4_window7_224": [0.982676, 0.987294, 0.986416, 0.990277, 0.991058],
}


class MultiViewDataset(Dataset):
    def __init__(self, df, views, transform):
        self.df = df.reset_index(drop=True)
        self.views = views
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        view_tensors = []
        for name, method in self.views:
            img_v = image if method is None else image.transpose(method)
            view_tensors.append(self.transform(img_v))
        return torch.stack(view_tensors), int(row["label"])


def fold_level_ensemble_macro_f1_flips_only(manifest_df, backbones, num_classes, cfg, device=None):
    """Same fold-loop and checkpoint-loading structure as the real
    fold_level_ensemble_macro_f1 in evaluate.py, with use_tta=True replaced
    by explicit flips-only view averaging."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_scores = []
    for fold in sorted(manifest_df.fold.unique()):
        if fold < 0:
            continue
        fold_models = []
        for backbone in backbones:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"{backbone}_fold{fold}.pth")
            ckpt = torch.load(ckpt_path, map_location=device)
            m = build_model(backbone, num_classes)
            m.load_state_dict(ckpt["model_state_dict"])
            fold_models.append(m.to(device).eval())

        val_df = manifest_df[(manifest_df.fold == fold) & (manifest_df.split == "val")]
        transform = MultiScaleTransform(cfg.img_size, is_train=False)
        ds = MultiViewDataset(val_df, FLIPS_ONLY_VIEWS, transform)
        loader = DataLoader(ds, batch_size=max(1, cfg.batch_size // len(FLIPS_ONLY_VIEWS)),
                             shuffle=False, num_workers=2)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for view_batches, labels in loader:
                B, n_views = view_batches.shape[0], view_batches.shape[1]
                flat = view_batches.view(B * n_views, *view_batches.shape[2:]).to(device)

                summed_probs = None
                for model in fold_models:
                    logits = model(flat)
                    probs = F.softmax(logits, dim=1)
                    probs = probs.view(B, n_views, -1).mean(dim=1)
                    summed_probs = probs if summed_probs is None else summed_probs + probs
                ensemble_probs = summed_probs / len(fold_models)

                all_preds.extend(ensemble_probs.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.numpy())

        fold_f1 = f1_score(all_labels, all_preds, average="macro")
        fold_scores.append(fold_f1)
        print(f"[fold_level_ensemble, flips-only TTA] fold {fold}: macro_f1={fold_f1:.4f}")

        del loader, ds, fold_models
        import gc
        gc.collect()
    return fold_scores


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(f"{CFG.manifest_dir}/manifest.csv")
    backbones = list(BACKBONE_FOLD_SCORES.keys())

    print("Recomputing fold-level ensemble scores under flips-only TTA...")
    ensemble_scores = fold_level_ensemble_macro_f1_flips_only(
        manifest, backbones, len(CFG.class_names), CFG, device
    )
    print(f"\nFold-level ensemble macro-F1 (flips-only TTA): "
          f"{[f'{s:.4f}' for s in ensemble_scores]}")
    print(f"Mean: {np.mean(ensemble_scores):.4f}, s.d.: {np.std(ensemble_scores, ddof=1):.4f}")

    results = []
    for backbone, backbone_scores in BACKBONE_FOLD_SCORES.items():
        t_stat, t_p = ttest_rel(ensemble_scores, backbone_scores)
        w_stat, w_p = wilcoxon(ensemble_scores, backbone_scores)
        diff = np.array(ensemble_scores) - np.array(backbone_scores)
        cohens_d = diff.mean() / diff.std(ddof=1)

        print(f"\nCashewNet (flips-only TTA) vs. {backbone}:")
        print(f"  t-statistic={t_stat:.3f}, t p-value={t_p:.4f}")
        print(f"  Wilcoxon statistic={w_stat:.3f}, Wilcoxon p-value={w_p:.4f}")
        print(f"  Cohen's d={cohens_d:.3f}")

        results.append({
            "comparison": f"CashewNet (flips-only TTA) vs. {backbone}",
            "t_statistic": t_stat, "t_pvalue": t_p,
            "wilcoxon_statistic": w_stat, "wilcoxon_pvalue": w_p,
            "cohens_d": cohens_d,
        })

    pd.DataFrame(results).to_csv(f"{OUT_DIR}/section43_flips_only_tta_significance.csv", index=False)
    print(f"\nSaved {OUT_DIR}/section43_flips_only_tta_significance.csv ")


if __name__ == "__main__":
    main()
