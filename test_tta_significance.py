"""
Determines whether flips-only TTA's accuracy advantage over full-6-view TTA
(99.59% vs. 99.32%, same held-out test set) is statistically meaningful,
using McNemar's test -- the correct paired test here, since both
configurations are evaluated on the identical 2,212 test images: what
matters is not their independent accuracy levels, but how often each
config gets right what the other gets wrong (the discordant pairs), not
just their marginal correct/incorrect counts.

Run from the project root, inside the container:
    python3 test_tta_significance.py
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


def mcnemar_exact(a_only, b_only):
    """Exact McNemar test via binomial test on the discordant pairs,
    avoiding a statsmodels dependency. Verified to exactly match
    statsmodels.stats.contingency_tables.mcnemar(exact=True) on multiple
    hand-constructed cases (e.g. 30 vs 10 discordant -> p=0.002221 both
    ways; 6 vs 0 -> p=0.0312 both ways)."""
    n = a_only + b_only
    if n == 0:
        return float("nan")
    k = min(a_only, b_only)
    return binomtest(k, n, 0.5, alternative="two-sided").pvalue

VIEW_SETS = {
    "flips_only": [
        ("identity", None),
        ("hflip", Image.FLIP_LEFT_RIGHT),
        ("vflip", Image.FLIP_TOP_BOTTOM),
    ],
    "full_6view": [
        ("identity", None),
        ("hflip", Image.FLIP_LEFT_RIGHT),
        ("vflip", Image.FLIP_TOP_BOTTOM),
        ("rot90", Image.ROTATE_90),
        ("rot180", Image.ROTATE_180),
        ("rot270", Image.ROTATE_270),
    ],
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


def get_per_image_correctness(models, df, view_set_name, device):
    """Returns a boolean array, one entry per test image IN THE SAME ORDER
    AS df, indicating whether that configuration predicted it correctly --
    the row order is what makes the later pairing valid."""
    views = VIEW_SETS[view_set_name]
    transform = MultiScaleTransform(CFG.img_size, is_train=False)
    ds = MultiViewDataset(df, views, transform)
    loader = DataLoader(ds, batch_size=max(1, CFG.batch_size // len(views)), shuffle=False, num_workers=2)

    all_correct = []
    with torch.no_grad():
        for view_batches, labels in loader:
            B, n_views = view_batches.shape[0], view_batches.shape[1]
            flat = view_batches.view(B * n_views, *view_batches.shape[2:]).to(device)

            summed_probs = None
            for model in models:
                logits = model(flat)
                probs = F.softmax(logits, dim=1)
                probs = probs.view(B, n_views, -1).mean(dim=1)
                summed_probs = probs if summed_probs is None else summed_probs + probs
            ensemble_probs = summed_probs / len(models)

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
    test_df = manifest[manifest.split == "test"].reset_index(drop=True)  # fixed order, used for BOTH configs

    best_models = torch.load(f"{CFG.checkpoint_dir}/best_models.pt", weights_only=False)
    models = [m.to(device).eval() for m in best_models.values()]

    print("Evaluating flips_only, saving per-image correctness...")
    correct_flips = get_per_image_correctness(models, test_df, "flips_only", device)
    print(f"  accuracy: {correct_flips.mean()*100:.2f}%")

    print("Evaluating full_6view, saving per-image correctness...")
    correct_full = get_per_image_correctness(models, test_df, "full_6view", device)
    print(f"  accuracy: {correct_full.mean()*100:.2f}%")

    # 2x2 contingency table of discordant/concordant pairs
    both_correct = int(np.sum(correct_flips & correct_full))
    flips_only_correct = int(np.sum(correct_flips & ~correct_full))   # flips right, full wrong
    full_only_correct = int(np.sum(~correct_flips & correct_full))    # full right, flips wrong
    both_wrong = int(np.sum(~correct_flips & ~correct_full))

    table = [[both_correct, flips_only_correct],
             [full_only_correct, both_wrong]]

    print("\n2x2 contingency table:")
    print(f"                          full_6view correct   full_6view wrong")
    print(f"  flips_only correct      {both_correct:>18}   {flips_only_correct:>16}")
    print(f"  flips_only wrong        {full_only_correct:>18}   {both_wrong:>16}")

    print(f"\nDiscordant pairs: flips-right-full-wrong={flips_only_correct}, "
          f"full-right-flips-wrong={full_only_correct}")

    if flips_only_correct + full_only_correct < 10:
        print(f"\nWARNING: only {flips_only_correct + full_only_correct} discordant pairs total. "
              f"McNemar's test (like the Wilcoxon test used) "
              f"has limited power with this few discordant observations -- a non-significant result "
              f"here would be consistent with either a genuinely small/absent difference OR an "
              f"underpowered test, not necessarily proof of no difference.")

    pvalue = mcnemar_exact(flips_only_correct, full_only_correct)
    print(f"\nMcNemar's exact test: p-value={pvalue:.4f} "
          f"(discordant pairs: {flips_only_correct} vs {full_only_correct})")

    pd.DataFrame([{
        "both_correct": both_correct, "flips_only_correct": flips_only_correct,
        "full_only_correct": full_only_correct, "both_wrong": both_wrong,
        "mcnemar_pvalue": pvalue,
        "flips_accuracy_pct": correct_flips.mean() * 100,
        "full_accuracy_pct": correct_full.mean() * 100,
    }]).to_csv(f"{OUT_DIR}/tta_mcnemar_test.csv", index=False)
    print(f"\nSaved {OUT_DIR}/tta_mcnemar_test.csv")


if __name__ == "__main__":
    main()
