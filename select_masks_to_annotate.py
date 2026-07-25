"""
Randomly (but reproducibly) selects a target number of test-split images per
class to annotate, and copies them into a flat staging folder so they're
easy to load into an annotation tool (CVAT/Labelme) without hunting through
the full dataset tree.

Run from the project root (same place manifest.csv lives):
    python3 select_masks_to_annotate.py
"""
import pandas as pd
import numpy as np
import os
import shutil

SEED = 42
CLASS_NAMES = ["anthracnose", "healthy", "leaf_miner", "red_rust"]

# healthy has no lesion to annotate, so it's given a token count only in
# case you want a few empty-mask baseline images; drop to 0 if not needed.
TARGET_PER_CLASS = {
    "anthracnose": 30,
    "healthy": 10,
    "leaf_miner": 50,
    "red_rust": 30,
}

STAGING_DIR = "annotation_staging"


def main():
    manifest = pd.read_csv("manifests/manifest.csv")
    test_df = manifest[manifest.split == "test"]
    rng = np.random.RandomState(SEED)

    os.makedirs(STAGING_DIR, exist_ok=True)
    selection_rows = []

    for cls_idx, cls in enumerate(CLASS_NAMES):
        target = TARGET_PER_CLASS.get(cls, 0)
        if target == 0:
            continue
        cls_paths = test_df[test_df.label == cls_idx]["path"].tolist()
        n = min(target, len(cls_paths))
        chosen = rng.choice(cls_paths, size=n, replace=False)

        cls_stage_dir = os.path.join(STAGING_DIR, cls)
        os.makedirs(cls_stage_dir, exist_ok=True)
        for src_path in chosen:
            fname = os.path.basename(src_path)
            dst_path = os.path.join(cls_stage_dir, fname)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
            selection_rows.append({"class": cls, "source_path": src_path,
                                    "staged_path": dst_path,
                                    "mask_should_go_to": f"lesion_masks/{cls}/"
                                    f"{os.path.splitext(fname)[0]}.png"})

    sel_df = pd.DataFrame(selection_rows)
    sel_df.to_csv("annotation_selection_manifest.csv", index=False)

    print(f"Staged {len(sel_df)} images into {STAGING_DIR}/<class>/ for annotation.")
    print(sel_df.groupby("class").size())
    print(f"\nSee annotation_selection_manifest.csv for exactly where each "
          f"mask needs to be saved once you're done.")


if __name__ == "__main__":
    main()
