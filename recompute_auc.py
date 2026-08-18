"""
Closes a gap in recompute_4_flips_tta.py, which computed accuracy and
macro F1 under flips-only TTA but never computed macro AUC -- meaning no
verified AUC figure currently exists for this configuration. This script
adds that, using the same probability outputs and bootstrap protocol
(1,000 resamples), so the AUC figure is directly
comparable to the accuracy/F1 figures already established.

Run from the project root, inside the container:
    python3 recompute_auc.py
"""
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

from config import CFG
from datasets import MultiScaleTransform

OUT_DIR = CFG.output_dir
N_BOOTSTRAP = 1000

FLIPS_ONLY_VIEWS = [
    ("identity", None),
    ("hflip", Image.FLIP_LEFT_RIGHT),
    ("vflip", Image.FLIP_TOP_BOTTOM),
]


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


def get_flips_only_probs(model_list, df, device):
    transform = MultiScaleTransform(CFG.img_size, is_train=False)
    ds = MultiViewDataset(df, FLIPS_ONLY_VIEWS, transform)
    loader = DataLoader(ds, batch_size=max(1, CFG.batch_size // len(FLIPS_ONLY_VIEWS)),
                         shuffle=False, num_workers=2)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for view_batches, labels in loader:
            B, n_views = view_batches.shape[0], view_batches.shape[1]
            flat = view_batches.view(B * n_views, *view_batches.shape[2:]).to(device)

            summed_probs = None
            for model in model_list:
                logits = model(flat)
                probs = F.softmax(logits, dim=1)
                probs = probs.view(B, n_views, -1).mean(dim=1)
                summed_probs = probs if summed_probs is None else summed_probs + probs
            ensemble_probs = summed_probs / len(model_list)

            all_probs.append(ensemble_probs.cpu())
            all_labels.append(labels)

    probs_cat = torch.cat(all_probs).numpy()
    labels_cat = torch.cat(all_labels).numpy()
    del loader, ds
    import gc
    gc.collect()
    return probs_cat, labels_cat


def macro_auc_ovr(probs, labels, n_classes):
    labels_bin = label_binarize(labels, classes=list(range(n_classes)))
    return roc_auc_score(labels_bin, probs, average="macro", multi_class="ovr")


def bootstrap_auc_ci(probs, labels, n_classes, n_bootstrap=N_BOOTSTRAP, seed=42):
    rng = np.random.RandomState(seed)
    n = len(labels)
    point_auc = macro_auc_ovr(probs, labels, n_classes)

    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_probs, boot_labels = probs[idx], labels[idx]
        # a resample might not contain all classes; skip those resamples
        # rather than let roc_auc_score error out, same convention as
        # scikit-learn's own handling of degenerate bootstrap folds.
        if len(np.unique(boot_labels)) < n_classes:
            continue
        try:
            boot_aucs.append(macro_auc_ovr(boot_probs, boot_labels, n_classes))
        except ValueError:
            continue

    lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
    return point_auc, lo, hi, len(boot_aucs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(f"{CFG.manifest_dir}/manifest.csv")
    test_df = manifest[manifest.split == "test"].reset_index(drop=True)
    n_classes = len(CFG.class_names)

    best_models = torch.load(f"{CFG.checkpoint_dir}/best_models.pt", weights_only=False)
    model_dict = {name: m.to(device).eval() for name, m in best_models.items()}
    all_models = list(model_dict.values())

    results = {}
    for name, model in model_dict.items():
        print(f"Evaluating {name} alone, macro AUC under flips-only TTA...")
        probs, labels = get_flips_only_probs([model], test_df, device)
        auc, lo, hi, n_valid = bootstrap_auc_ci(probs, labels, n_classes)
        print(f"  macro AUC={auc:.4f} (95% CI {lo:.4f}-{hi:.4f}, {n_valid}/{N_BOOTSTRAP} valid resamples)")
        results[name] = {"macro_auc": auc, "auc_ci_lo": lo, "auc_ci_hi": hi}

    print("Evaluating 3-backbone ensemble, macro AUC under flips-only TTA...")
    probs, labels = get_flips_only_probs(all_models, test_df, device)
    auc, lo, hi, n_valid = bootstrap_auc_ci(probs, labels, n_classes)
    print(f"  macro AUC={auc:.4f} (95% CI {lo:.4f}-{hi:.4f}, {n_valid}/{N_BOOTSTRAP} valid resamples)")
    results["CashewNet (ensemble)"] = {"macro_auc": auc, "auc_ci_lo": lo, "auc_ci_hi": hi}

    pd.DataFrame(results).T.to_csv(f"{OUT_DIR}/flips_only_tta_auc.csv")
    print(f"\nSaved {OUT_DIR}/flips_only_tta_auc.csv")


if __name__ == "__main__":
    main()
