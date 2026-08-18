"""
Recomputes flips-only TTA: each individual backbone's
own accuracy/macro-F1, plus the 3-backbone ensemble
(already established: 99.59%/0.9960), each with a FRESH bootstrap 95% CI
computed specifically for this configuration.

Run from the project root, inside the container:
    python3 recompute_table4_flips_tta.py
"""
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

from config import CFG
from datasets import MultiScaleTransform

OUT_DIR = CFG.output_dir
N_BOOTSTRAP = 1000  # matches the resample count already used

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


def get_flips_only_probs(models, df, device, single_model_only=None):
    """Returns (probs, labels) arrays for the full test set under
    flips-only TTA. If single_model_only is given (a single model, not a
    list), computes that ONE backbone's own result rather than the
    ensemble -- reuses identical view-generation/averaging logic in both
    cases so backbone-alone and ensemble numbers are directly comparable,
    computed the same way."""
    model_list = [single_model_only] if single_model_only is not None else models
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


def bootstrap_ci(probs, labels, n_classes, n_bootstrap=N_BOOTSTRAP, seed=42):
    """Percentile bootstrap over individual test predictions, resampling
    images with replacement -- matches the protocol already described 
    for the original CI (1,000 resamples)."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    preds = probs.argmax(axis=1)

    point_accuracy = (preds == labels).mean()
    point_f1 = f1_score(labels, preds, average="macro")

    boot_accuracies, boot_f1s = [], []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_preds = preds[idx]
        boot_labels = labels[idx]
        boot_accuracies.append((boot_preds == boot_labels).mean())
        boot_f1s.append(f1_score(boot_labels, boot_preds, average="macro", zero_division=0))

    acc_lo, acc_hi = np.percentile(boot_accuracies, [2.5, 97.5])
    f1_lo, f1_hi = np.percentile(boot_f1s, [2.5, 97.5])

    return {
        "accuracy_pct": point_accuracy * 100,
        "accuracy_ci_lo": acc_lo * 100, "accuracy_ci_hi": acc_hi * 100,
        "macro_f1": point_f1,
        "macro_f1_ci_lo": f1_lo, "macro_f1_ci_hi": f1_hi,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(f"{CFG.manifest_dir}/manifest.csv")
    test_df = manifest[manifest.split == "test"].reset_index(drop=True)
    n_classes = len(CFG.class_names)

    best_models = torch.load(f"{CFG.checkpoint_dir}/best_models.pt", weights_only=False)
    model_dict = {name: m.to(device).eval() for name, m in best_models.items()}
    models = list(model_dict.values())

    results = {}

    for name, model in model_dict.items():
        print(f"Evaluating {name} alone under flips-only TTA...")
        probs, labels = get_flips_only_probs(models, test_df, device, single_model_only=model)
        stats = bootstrap_ci(probs, labels, n_classes)
        results[name] = stats
        print(f"  accuracy={stats['accuracy_pct']:.2f}% "
              f"(95% CI {stats['accuracy_ci_lo']:.2f}-{stats['accuracy_ci_hi']:.2f}%), "
              f"macro_f1={stats['macro_f1']:.4f} "
              f"(95% CI {stats['macro_f1_ci_lo']:.4f}-{stats['macro_f1_ci_hi']:.4f})")

    print("Evaluating 3-backbone ensemble under flips-only TTA...")
    probs, labels = get_flips_only_probs(models, test_df, device, single_model_only=None)
    stats = bootstrap_ci(probs, labels, n_classes)
    results["CashewNet (ensemble)"] = stats
    print(f"  accuracy={stats['accuracy_pct']:.2f}% "
          f"(95% CI {stats['accuracy_ci_lo']:.2f}-{stats['accuracy_ci_hi']:.2f}%), "
          f"macro_f1={stats['macro_f1']:.4f} "
          f"(95% CI {stats['macro_f1_ci_lo']:.4f}-{stats['macro_f1_ci_hi']:.4f})")
    print("  (cross-check: point accuracy should be close to the already-established 99.59%)")

    out_df = pd.DataFrame(results).T
    out_df.to_csv(f"{OUT_DIR}/flips_only_tta.csv")
    print(f"\nSaved {OUT_DIR}/flips_only_tta.csv, "
          f"replacing the old 6-view-TTA-based numbers.")


if __name__ == "__main__":
    main()
