"""
Reports per-fold class distribution to quantify how much class
imbalance the group-aware splitting constraint introduces
across folds -- pure aggregation of the existing manifest, no retraining
or new inference needed.

IMPORTANT: this manifest is long-format -- every train/val-pool image
appears once per fold (5 rows per image total), with the 'split' column
indicating whether that image is 'train' or 'val' WITHIN that specific
fold's context (confirmed via diagnose_fold_column.py: 12,570 unique
images x 5 folds = 62,850 total rows; val totals 12,570 = 5 x 2,514
per-fold validation set size). Aggregating on 'fold' alone (an earlier,
incorrect version of this script did this) counts every image under every
fold regardless of train/val role, which trivially reproduces the full
pool total for every fold and does not measure per-fold distribution at
all. The correct aggregation filters to each fold's own validation subset.

Run from the project root:
    python3 report_fold_class_distribution.py
"""
import pandas as pd

from config import CFG


def main():
    manifest = pd.read_csv(f"{CFG.manifest_dir}/manifest.csv")

    # Each fold's held-out VALIDATION set specifically -- not the full
    # per-fold row count, which includes that fold's training rows too.
    val_rows = manifest[manifest["split"] == "val"].copy()

    counts = pd.crosstab(val_rows["fold"], val_rows["label"])
    counts.columns = [CFG.class_names[c] for c in counts.columns]
    print("Per-fold VALIDATION set class counts:")
    print(counts.to_string())
    counts.to_csv(f"{CFG.output_dir}/fold_class_distribution.csv")

    print("\nPer-class range and standard deviation across the 5 folds' validation sets:")
    summary_rows = []
    for cls in counts.columns:
        vals = counts[cls]
        summary_rows.append({
            "class": cls,
            "min": vals.min(),
            "max": vals.max(),
            "range": vals.max() - vals.min(),
            "mean": vals.mean(),
            "std": vals.std(),
            "cv_pct": 100 * vals.std() / vals.mean(),
        })
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    summary.to_csv(f"{CFG.output_dir}/fold_class_distribution_summary.csv", index=False)
    print(f"\nSaved {CFG.output_dir}/fold_class_distribution.csv and "
          f"{CFG.output_dir}/fold_class_distribution_summary.csv")
    print("\nA low coefficient of variation (cv_pct, typically under ~10-15% for a "
          "reasonably balanced split) indicates the group-aware constraint "
          "did not introduce meaningful additional class imbalance beyond the dataset's "
          "own natural class proportions across folds' validation sets.")


if __name__ == "__main__":
    main()
