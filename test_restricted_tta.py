"""
Tests whether the ensemble's TTA underperformance (99.32% with full 6-view
TTA vs. 99.46% without TTA) is explained by a train/test augmentation
mismatch: training used mild +/-15 degree rotation, but the
6-view TTA scheme includes 90/180/270 degree rotations that
training never prepared the model for. This script evaluates a restricted
3-view TTA scheme -- identity, horizontal flip, vertical flip only, i.e.
exactly the augmentation types (not full rotation) present in training --
to see whether removing the large-rotation views recovers or exceeds the
no-TTA accuracy.

Does NOT rely on EnsembleModel.predict_batch's built-in use_tta flag, since
that applies the fixed 6-view scheme; this script implements the geometric
views directly so an arbitrary subset can be tested.

Run from the project root, inside the container:
    python3 test_restricted_tta.py
"""
import pandas as pd
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")  # BUGFIX: avoids "Too many
# open files" -- the default 'file_descriptor' sharing strategy accumulates open FDs
# across each DataLoader's worker processes, and creating multiple DataLoaders in
# sequence (once per view-set below) can exhaust the OS limit before cleanup catches
# up, especially for the view-set with the most iterations (full_6view, which has the
# smallest per-batch size given the batch_size // n_views calculation below).
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from config import CFG
from datasets import MultiScaleTransform

OUT_DIR = CFG.output_dir

# Each entry: (name, PIL transpose method or None for identity)
VIEW_SETS = {
    "identity_only": [("identity", None)],
    "flips_only (matches training augmentation)": [
        ("identity", None),
        ("hflip", Image.FLIP_LEFT_RIGHT),
        ("vflip", Image.FLIP_TOP_BOTTOM),
    ],
    "full_6view (existing TTA scheme)": [
        ("identity", None),
        ("hflip", Image.FLIP_LEFT_RIGHT),
        ("vflip", Image.FLIP_TOP_BOTTOM),
        ("rot90", Image.ROTATE_90),
        ("rot180", Image.ROTATE_180),
        ("rot270", Image.ROTATE_270),
    ],
}


class MultiViewDataset(Dataset):
    """Returns all views of a single image, pre-transformed, so a single
    DataLoader batch corresponds to one image's full view set -- keeps the
    per-image softmax averaging simple and correct."""
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


def evaluate_view_set(models, df, view_set_name, device):
    views = VIEW_SETS[view_set_name]
    transform = MultiScaleTransform(CFG.img_size, is_train=False)
    ds = MultiViewDataset(df, views, transform)
    loader = DataLoader(ds, batch_size=max(1, CFG.batch_size // len(views)), shuffle=False, num_workers=2)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for view_batches, labels in loader:
            # view_batches: (B, n_views, C, H, W)
            B, n_views = view_batches.shape[0], view_batches.shape[1]
            flat = view_batches.view(B * n_views, *view_batches.shape[2:]).to(device)

            summed_probs = None
            for model in models:
                logits = model(flat)
                probs = F.softmax(logits, dim=1)
                probs = probs.view(B, n_views, -1).mean(dim=1)  # average views, per backbone
                summed_probs = probs if summed_probs is None else summed_probs + probs
            ensemble_probs = summed_probs / len(models)  # average backbones

            preds = ensemble_probs.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    preds_cat = torch.cat(all_preds)
    labels_cat = torch.cat(all_labels)
    accuracy = (preds_cat == labels_cat).float().mean().item()

    # Explicit cleanup: force DataLoader worker processes (and their open file
    # handles) to be torn down before the next view-set constructs a new
    # DataLoader, rather than relying on garbage collection timing.
    del loader, ds
    import gc
    gc.collect()

    f1s = []
    for c in range(len(CFG.class_names)):
        tp = ((preds_cat == c) & (labels_cat == c)).sum().item()
        fp = ((preds_cat == c) & (labels_cat != c)).sum().item()
        fn = ((preds_cat != c) & (labels_cat == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s)
    return accuracy, macro_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(f"{CFG.manifest_dir}/manifest.csv")
    test_df = manifest[manifest.split == "test"].reset_index(drop=True)

    best_models = torch.load(f"{CFG.checkpoint_dir}/best_models.pt", weights_only=False)
    models = [m.to(device).eval() for m in best_models.values()]

    results = []
    for view_set_name in VIEW_SETS:
        print(f"Evaluating '{view_set_name}' ({len(VIEW_SETS[view_set_name])} view(s))...")
        acc, f1 = evaluate_view_set(models, test_df, view_set_name, device)
        print(f"  accuracy={acc*100:.2f}%  macro_f1={f1:.4f}")
        results.append({"view_set": view_set_name, "n_views": len(VIEW_SETS[view_set_name]),
                         "accuracy_pct": acc * 100, "macro_f1": f1})

    df = pd.DataFrame(results)
    df.to_csv(f"{OUT_DIR}/restricted_tta_comparison.csv", index=False)
    print(f"\nSaved {OUT_DIR}/restricted_tta_comparison.csv")

    identity_row = df[df.view_set == "identity_only"].iloc[0]
    flips_row = df[df.view_set.str.contains("flips_only")].iloc[0]
    full_row = df[df.view_set.str.contains("full_6view")].iloc[0]

    print("\n--- Interpretation ---")
    print(f"identity only:        {identity_row['accuracy_pct']:.2f}%  (should be close to the earlier "
          f"'ensemble, no TTA' result of 99.46% -- cross-check)")
    print(f"flips only (3-view):  {flips_row['accuracy_pct']:.2f}%")
    print(f"full 6-view (existing TTA): {full_row['accuracy_pct']:.2f}%  (should match the reported 99.32%)")
    if flips_row["accuracy_pct"] >= identity_row["accuracy_pct"]:
        print("\nRestricting TTA to flips-only (matching training augmentation) performed AT LEAST AS WELL "
              "as no TTA at all -- consistent with the hypothesis that the large-rotation views specifically "
              "are responsible for full 6-view TTA's underperformance, not TTA/averaging in general.")
    else:
        print("\nFlips-only TTA did NOT recover the no-TTA result -- the augmentation-mismatch hypothesis "
              "is not fully sufficient on its own; some other factor may also be contributing.")


if __name__ == "__main__":
    main()
