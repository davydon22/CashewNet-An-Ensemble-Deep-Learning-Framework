"""
Tests directly whether the ensemble's accuracy under flips-only TTA
(99.59%) is significantly different from ConvNeXt-Tiny alone under the
same flips-only TTA (99.41%) -- the point estimates now show a gap where
the old 6-view-TTA scheme showed an exact tie, but their bootstrap CIs
still overlap, so this needs a proper paired test (same images, same
configuration comparison) rather than being inferred from overlapping
independent CIs.

Run from the project root, inside the container:
    python3 test_ensemble_vs_convnext_significance.py
"""
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from scipy.stats import binomtest

from config import CFG
from datasets import MultiScaleTransform

OUT_DIR = CFG.output_dir

FLIPS_ONLY_VIEWS = [
    ("identity", None),
    ("hflip", Image.FLIP_LEFT_RIGHT),
    ("vflip", Image.FLIP_TOP_BOTTOM),
]


def mcnemar_exact(a_only, b_only):
    """Verified in earlier scripts to exactly match
    statsmodels.stats.contingency_tables.mcnemar(exact=True)."""
    n = a_only + b_only
    if n == 0:
        return float("nan")
    k = min(a_only, b_only)
    return binomtest(k, n, 0.5, alternative="two-sided").pvalue


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


def get_per_image_correctness(model_list, df, device):
    """model_list: pass [convnext_model] for ConvNeXt-alone, or all three
    models for the ensemble -- same averaging logic either way, so results
    are directly comparable."""
    transform = MultiScaleTransform(CFG.img_size, is_train=False)
    ds = MultiViewDataset(df, FLIPS_ONLY_VIEWS, transform)
    loader = DataLoader(ds, batch_size=max(1, CFG.batch_size // len(FLIPS_ONLY_VIEWS)),
                         shuffle=False, num_workers=2)

    all_correct = []
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

            preds = ensemble_probs.argmax(dim=1).cpu()
            all_correct.append(preds == labels)

    result = torch.cat(all_correct).numpy()
    del loader, ds
    import gc
    gc.collect()
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(f"{CFG.manifest_dir}/manifest.csv")
    test_df = manifest[manifest.split == "test"].reset_index(drop=True)

    best_models = torch.load(f"{CFG.checkpoint_dir}/best_models.pt", weights_only=False)
    model_dict = {name: m.to(device).eval() for name, m in best_models.items()}
    all_models = list(model_dict.values())
    convnext_model = model_dict["convnext_tiny"]

    print("Evaluating ConvNeXt-Tiny alone, flips-only TTA...")
    correct_convnext = get_per_image_correctness([convnext_model], test_df, device)
    print(f"  accuracy: {correct_convnext.mean()*100:.2f}% (cross-check vs. 99.41%)")

    print("Evaluating 3-backbone ensemble, flips-only TTA...")
    correct_ensemble = get_per_image_correctness(all_models, test_df, device)
    print(f"  accuracy: {correct_ensemble.mean()*100:.2f}% (cross-check vs. 99.59%)")

    both_correct = int(np.sum(correct_ensemble & correct_convnext))
    ensemble_only = int(np.sum(correct_ensemble & ~correct_convnext))
    convnext_only = int(np.sum(~correct_ensemble & correct_convnext))
    both_wrong = int(np.sum(~correct_ensemble & ~correct_convnext))

    print("\n2x2 contingency table:")
    print(f"                         ConvNeXt correct   ConvNeXt wrong")
    print(f"  Ensemble correct       {both_correct:>17}   {ensemble_only:>14}")
    print(f"  Ensemble wrong         {convnext_only:>17}   {both_wrong:>14}")

    print(f"\nDiscordant pairs: ensemble-right-convnext-wrong={ensemble_only}, "
          f"convnext-right-ensemble-wrong={convnext_only}")

    if ensemble_only + convnext_only < 10:
        print(f"\nWARNING: only {ensemble_only + convnext_only} discordant pairs total -- "
              f"limited power, same caveat as every other paired test in this study at this "
              f"sample size. A non-significant result would be consistent with either a "
              f"genuinely small difference or an underpowered test.")

    pvalue = mcnemar_exact(ensemble_only, convnext_only)
    print(f"\nMcNemar's exact test: p-value={pvalue:.4f}")

    if pvalue < 0.05:
        print("\nThe ensemble is SIGNIFICANTLY better than ConvNeXt-Tiny alone under flips-only "
              "TTA -- the 'ConvNeXt-Tiny ties the ensemble' claim should be revised: this holds "
              "for the old 6-view-TTA scheme but not for the new flips-only configuration.")
    else:
        print("\nNo significant difference detected at this sample size -- the point-estimate gap "
              "(99.59% vs 99.41%) is directionally consistent with a real but small ensemble "
              "advantage, but is not independently confirmed here. The 'ConvNeXt-Tiny remains "
              "competitive with the ensemble' framing can reasonably be retained, with the smaller "
              "point-estimate gap noted rather than treated as a confirmed effect.")

    pd.DataFrame([{
        "both_correct": both_correct, "ensemble_only_correct": ensemble_only,
        "convnext_only_correct": convnext_only, "both_wrong": both_wrong,
        "mcnemar_pvalue": pvalue,
        "ensemble_accuracy_pct": correct_ensemble.mean() * 100,
        "convnext_accuracy_pct": correct_convnext.mean() * 100,
    }]).to_csv(f"{OUT_DIR}/ensemble_vs_convnext_flips_tta_significance.csv", index=False)
    print(f"\nSaved {OUT_DIR}/ensemble_vs_convnext_flips_tta_significance.csv")


if __name__ == "__main__":
    main()
