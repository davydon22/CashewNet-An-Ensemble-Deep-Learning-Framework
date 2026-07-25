"""
Dataset integrity audit and group-aware splitting.

This script does three things:
  1. Perceptual-hashes every image in the raw pool and clusters near-duplicates.
  2. Splits the pool into train / test (once) and train / 5-fold CV, guaranteeing
     every image in a duplicate cluster stays entirely within one split
     (StratifiedGroupKFold-style logic implemented via GroupShuffleSplit).
  3. Writes manifests (image_path,label,group_id,split) and a background/occlusion
     complexity report per class, so both concerns are addressed with numbers,
     not assumptions.

Run once, before any training. If duplicate clusters span your *original*
train/val/test folders, that is direct evidence of the leakage risk.
"""
import os
import json
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

try:
    import imagehash
except ImportError:
    raise SystemExit(
        "Run: pip install imagehash --break-system-packages"
    )

from config import CFG


# ---------------------------------------------------------------------
# 1. Perceptual hashing + duplicate clustering
# ---------------------------------------------------------------------
def build_hash_index(root):
    """Return {path: (phash, class_name)} for every image under root/<class>/*"""
    index = {}
    for cls in sorted(os.listdir(root)):
        cls_path = os.path.join(root, cls)
        if not os.path.isdir(cls_path) or cls.startswith('.'):
            continue
        for fname in os.listdir(cls_path):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            fpath = os.path.join(cls_path, fname)
            try:
                h = imagehash.phash(Image.open(fpath).convert("RGB"))
                index[fpath] = (h, cls)
            except Exception as e:
                print(f"⚠️ Skipping unreadable file {fpath}: {e}")
    return index


def cluster_duplicates(index, hamming_threshold):
    """
    Union-find style clustering: two images are the same cluster if their
    phash Hamming distance <= threshold. O(n^2) — fine up to a few tens of
    thousands of images; for larger pools, bucket by hash prefix first.
    """
    paths = list(index.keys())
    n = len(paths)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    hashes = [index[p][0] for p in paths]
    for i in range(n):
        for j in range(i + 1, n):
            if hashes[i] - hashes[j] <= hamming_threshold:
                union(i, j)

    clusters = defaultdict(list)
    for i, p in enumerate(paths):
        clusters[find(i)].append(p)

    # stable group id = md5 of sorted member list
    group_of = {}
    for members in clusters.values():
        gid = hashlib.md5("|".join(sorted(members)).encode()).hexdigest()[:12]
        for m in members:
            group_of[m] = gid

    return group_of, clusters


# ---------------------------------------------------------------------
# 2. Background / occlusion proxy stats (R2-#1)
# ---------------------------------------------------------------------
def background_complexity_stats(index):
    """
    Cheap, defensible proxies (not a substitute for expert annotation, but far
    better than the current qualitative-only description):
      - edge_density: fraction of Sobel-edge pixels (background clutter proxy)
      - green_fraction: fraction of pixels in a broad "leaf-green" HSV band
        (1 - green_fraction is a rough occlusion/background proxy)
    """
    import cv2
    rows = []
    for path, (h, cls) in index.items():
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (256, 256))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = edges.mean() / 255.0

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([25, 30, 30])
        upper = np.array([95, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        green_fraction = mask.mean() / 255.0

        rows.append({
            "path": path, "class": cls,
            "edge_density": edge_density,
            "green_fraction": green_fraction,
            "occlusion_proxy": 1 - green_fraction,
        })
    df = pd.DataFrame(rows)
    summary = df.groupby("class")[["edge_density", "green_fraction", "occlusion_proxy"]].agg(["mean", "std"])
    return df, summary


# ---------------------------------------------------------------------
# 3. Group-aware split (test held out once; remaining data k-folded)
# ---------------------------------------------------------------------
def group_aware_split(index, group_of, cfg):
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

    paths = np.array(list(index.keys()))
    labels = np.array([cfg.class_names.index(index[p][1]) for p in paths])
    groups = np.array([group_of[p] for p in paths])

    # Held-out test split (group-aware: whole duplicate clusters go to one side)
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_fraction, random_state=cfg.seed)
    trainval_idx, test_idx = next(gss.split(paths, labels, groups))

    records = []
    for i in test_idx:
        records.append({"path": paths[i], "label": labels[i], "class": cfg.class_names[labels[i]],
                         "group": groups[i], "split": "test", "fold": -1})

    # 5-fold (group-aware, stratified) over the remaining train+val pool
    trainval_paths = paths[trainval_idx]
    trainval_labels = labels[trainval_idx]
    trainval_groups = groups[trainval_idx]

    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    for fold, (tr_i, va_i) in enumerate(sgkf.split(trainval_paths, trainval_labels, trainval_groups)):
        for i in tr_i:
            records.append({"path": trainval_paths[i], "label": trainval_labels[i],
                             "class": cfg.class_names[trainval_labels[i]],
                             "group": trainval_groups[i], "split": "train", "fold": fold})
        for i in va_i:
            records.append({"path": trainval_paths[i], "label": trainval_labels[i],
                             "class": cfg.class_names[trainval_labels[i]],
                             "group": trainval_groups[i], "split": "val", "fold": fold})

    return pd.DataFrame.from_records(records)


def leakage_check(manifest_df):
    """Assert no group_id appears in both test and (train or val) — hard fail if so."""
    test_groups = set(manifest_df[manifest_df.split == "test"].group)
    other_groups = set(manifest_df[manifest_df.split != "test"].group)
    overlap = test_groups & other_groups
    if overlap:
        raise RuntimeError(
            f"LEAKAGE: {len(overlap)} duplicate-image groups appear in BOTH the "
            f"test split and train/val. Fix group_aware_split before proceeding."
        )
    print("✅ No duplicate-image group spans the test split and train/val — leakage check passed.")


def main():
    cfg = CFG
    cfg.ensure_dirs()

    print("🔎 Hashing all images...")
    index = build_hash_index(cfg.raw_dataset_root)
    print(f"   {len(index)} images found across {len(cfg.class_names)} classes")

    print("🔎 Clustering near-duplicates (Hamming <= "
          f"{cfg.phash_hamming_threshold})...")
    group_of, clusters = cluster_duplicates(index, cfg.phash_hamming_threshold)
    dup_clusters = {gid: members for gid, members in
                    defaultdict(list, {v: [] for v in group_of.values()}).items()}
    multi = [m for m in clusters.values() if len(m) > 1]
    print(f"   {sum(len(m) for m in multi)} images fall into {len(multi)} duplicate clusters "
          f"(cluster size > 1). These MUST stay within a single split.")

    print("🔎 Computing background/occlusion proxies...")
    stats_df, stats_summary = background_complexity_stats(index)
    stats_summary.to_csv(os.path.join(cfg.manifest_dir, "background_complexity_by_class.csv"))
    print(stats_summary)

    print("🔎 Building group-aware test / 5-fold split...")
    manifest = group_aware_split(index, group_of, cfg)
    leakage_check(manifest)

    manifest_path = os.path.join(cfg.manifest_dir, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"✅ Manifest written to {manifest_path}")

    # Per-class counts per split, for R3-#5
    counts = manifest.groupby(["split", "class"]).size().unstack(fill_value=0)
    counts.to_csv(os.path.join(cfg.manifest_dir, "class_counts_by_split.csv"))
    print(counts)

    with open(os.path.join(cfg.manifest_dir, "audit_summary.json"), "w") as f:
        json.dump({
            "n_images_total": len(index),
            "n_duplicate_clusters": len(multi),
            "n_images_in_duplicate_clusters": sum(len(m) for m in multi),
            "hamming_threshold": cfg.phash_hamming_threshold,
            "n_folds": cfg.n_folds,
            "test_fraction": cfg.test_fraction,
        }, f, indent=2)


if __name__ == "__main__":
    main()
