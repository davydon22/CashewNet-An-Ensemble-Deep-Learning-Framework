"""
Isolates the rotation views' specific contribution, completing the earlier
investigation. We've established: identity-only (no TTA) = 99.46%,
flips-only (identity+hflip+vflip) = 99.59% (significantly better, p=0.0312),
full-6-view (flips + 90/180/270 rotations) = 99.32% (significantly worse
than flips-only). This script tests "rotations-only" (identity + the three
large rotations, NO flips) directly against no-TTA, to determine whether
rotations specifically -- independent of whatever flips contribute -- are
responsible for the degradation, rather than some interaction between
flips and rotations together.

Run from the project root, inside the container:
    python3 test_rotation_isolation.py
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
    "identity_only": [
        ("identity", None),
    ],
    "rotations_only": [
        ("identity", None),
        ("rot90", Image.ROTATE_90),
        ("rot180", Image.ROTATE_180),
        ("rot270", Image.ROTATE_270),
    ],
}
CONFIG_A, CONFIG_B = "rotations_only", "identity_only"  # A=rotations, B=no-TTA baseline


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

    print(f"Evaluating {CONFIG_A} ({len(VIEW_SETS[CONFIG_A])} views), saving per-image correctness...")
    correct_a = get_per_image_correctness(models, test_df, CONFIG_A, device)
    print(f"  accuracy: {correct_a.mean()*100:.2f}%")

    print(f"Evaluating {CONFIG_B} ({len(VIEW_SETS[CONFIG_B])} views), saving per-image correctness...")
    correct_b = get_per_image_correctness(models, test_df, CONFIG_B, device)
    print(f"  accuracy: {correct_b.mean()*100:.2f}%")

    # 2x2 contingency table of discordant/concordant pairs
    both_correct = int(np.sum(correct_a & correct_b))
    a_only_correct = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    b_only_correct = int(np.sum(~correct_a & correct_b))   # B right, A wrong
    both_wrong = int(np.sum(~correct_a & ~correct_b))

    print("\n2x2 contingency table:")
    print(f"                     {CONFIG_B} correct   {CONFIG_B} wrong")
    print(f"  {CONFIG_A} correct   {both_correct:>18}   {a_only_correct:>16}")
    print(f"  {CONFIG_A} wrong     {b_only_correct:>18}   {both_wrong:>16}")

    print(f"\nDiscordant pairs: {CONFIG_A}-right-{CONFIG_B}-wrong={a_only_correct}, "
          f"{CONFIG_B}-right-{CONFIG_A}-wrong={b_only_correct}")

    if a_only_correct + b_only_correct < 10:
        print(f"\nWARNING: only {a_only_correct + b_only_correct} discordant pairs total. "
              f"McNemar's test (like the Wilcoxon test used) "
              f"has limited power with this few discordant observations -- a non-significant result "
              f"here would be consistent with either a genuinely small/absent difference OR an "
              f"underpowered test, not necessarily proof of no difference.")

    pvalue = mcnemar_exact(a_only_correct, b_only_correct)
    print(f"\nMcNemar's exact test: p-value={pvalue:.4f} "
          f"(discordant pairs: {a_only_correct} vs {b_only_correct})")

    if correct_a.mean() < correct_b.mean() and pvalue < 0.05:
        print(f"\n{CONFIG_A} is SIGNIFICANTLY WORSE than {CONFIG_B} -- this would confirm rotations "
              f"specifically (independent of flips) are responsible for degrading accuracy, not an "
              f"interaction between flips and rotations together.")
    elif correct_a.mean() < correct_b.mean():
        print(f"\n{CONFIG_A} is worse than {CONFIG_B} but NOT at conventional significance given "
              f"{a_only_correct + b_only_correct} discordant pairs -- directionally consistent with "
              f"rotations being the culprit, but not independently confirmed at this sample size.")
    else:
        print(f"\n{CONFIG_A} did NOT underperform {CONFIG_B} -- this would suggest rotations alone are "
              f"NOT sufficient to explain the full-6-view degradation, and the effect may depend on "
              f"combining rotations with flips specifically. Worth re-examining the earlier hypothesis.")

    pd.DataFrame([{
        "config_a": CONFIG_A, "config_b": CONFIG_B,
        "both_correct": both_correct, "a_only_correct": a_only_correct,
        "b_only_correct": b_only_correct, "both_wrong": both_wrong,
        "mcnemar_pvalue": pvalue,
        "a_accuracy_pct": correct_a.mean() * 100,
        "b_accuracy_pct": correct_b.mean() * 100,
    }]).to_csv(f"{OUT_DIR}/rotation_isolation_test.csv", index=False)
    print(f"\nSaved {OUT_DIR}/rotation_isolation_test.csv")


if __name__ == "__main__":
    main()

