"""
Diagnostic for the suspicious fold-distribution result: every fold showed
the EXACT total train/val pool count, which is only possible if either (a)
every row genuinely belongs to all 5 folds simultaneously (a real bug), or
(b) the 'fold' column doesn't mean "this row's held-out validation fold"
at all -- e.g. it's unpopulated, constant, or a leftover column from a
different pipeline stage.

Run from the project root:
    python3 diagnose_fold_column.py
"""
import pandas as pd
from config import CFG

manifest = pd.read_csv(f"{CFG.manifest_dir}/manifest.csv")
print("Manifest columns:", list(manifest.columns))
print()

train_val = manifest[manifest.split != "test"]
print(f"Rows in train/val pool: {len(train_val)}")
print()

print("Unique values in 'fold' column:", sorted(train_val["fold"].unique()))
print()

print("Row count per fold value (should sum to", len(train_val), "if mutually exclusive):")
print(train_val["fold"].value_counts().sort_index())
print("Sum of per-fold counts:", train_val["fold"].value_counts().sum())
print()

print("Sample of 10 random rows (path, label, fold):")
print(train_val[["path", "label", "fold"]].sample(10, random_state=0).to_string(index=False))
print()

# check whether 'fold' might actually be a duplicate-cluster id, or some
# other column, by checking its range and distinctness
print(f"fold column dtype: {train_val['fold'].dtype}")
print(f"fold column min/max: {train_val['fold'].min()} / {train_val['fold'].max()}")
print(f"Number of distinct fold values: {train_val['fold'].nunique()}")
print()

print("Unique values in 'split' column (whole manifest, not just train/val pool):")
print(manifest["split"].value_counts())
print()
print("The padding-control script filters on split == 'train' -- if that exact string "
      "isn't one of the values printed above, that filter silently matches zero rows "
      "for the WRONG reason (a naming mismatch, not a real empty result), and any rows "
      "it did get for fine-tuning are effectively the entire train/val pool regardless "
      "of the intended fold-0-holdout logic.")

