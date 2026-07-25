"""
Diagnoses WHY pointing_game_hit_rate might be persistently 0 by checking,
for each real (non-placeholder) annotated mask, how much of the lesion
survives the model's actual eval-time crop — separately from anything about
the model's attention itself.

Run from the project root:
    python3 diagnose_mask_alignment.py
"""
import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image
import torchvision.transforms.functional as TF

CLASS_NAMES = ["anthracnose", "healthy", "leaf_miner", "red_rust"]
MIN_LESION_FRACTION = 0.001


def main():
    sel = pd.read_csv("annotation_selection_manifest.csv")
    rows = []

    for _, row in sel.iterrows():
        mask_path = row["mask_should_go_to"]
        if not os.path.exists(mask_path):
            continue
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            continue
        raw_fraction = (raw_mask > 127).mean()
        if raw_fraction < MIN_LESION_FRACTION:
            continue  # skip placeholders, same as check_mask_progress.py

        orig_h, orig_w = raw_mask.shape

        # Apply the EXACT same eval-time transform as the model input
        mask_pil = Image.fromarray(raw_mask, mode="L")
        mask_pil = TF.resize(mask_pil, [256, 256], interpolation=TF.InterpolationMode.NEAREST)
        mask_pil = TF.center_crop(mask_pil, 224)
        cropped_mask = np.array(mask_pil) > 127
        cropped_fraction = cropped_mask.mean()

        rows.append({
            "class": row["class"],
            "orig_size": f"{orig_w}x{orig_h}",
            "raw_lesion_pct": raw_fraction * 100,
            "post_crop_lesion_pct": cropped_fraction * 100,
            "survived_crop": cropped_mask.sum() > 0,
            "pct_of_lesion_retained": (100 * cropped_mask.sum() / max(raw_mask[raw_mask>127].size, 1))
                                       if raw_mask[raw_mask>127].size > 0 else 0,
        })

    df = pd.DataFrame(rows)
    print("=" * 70)
    print(f"Checked {len(df)} real annotated masks")
    print("=" * 70)
    print(df.to_string(index=False))

    n_lost_entirely = (~df["survived_crop"]).sum()
    print(f"\n{n_lost_entirely}/{len(df)} masks have ZERO lesion pixels remaining "
          f"after the model's actual crop — meaning the model literally never "
          f"sees any part of the annotated lesion for these images, regardless "
          f"of how good its attention mechanism is.")

    print(f"\nImage dimensions found: {df['orig_size'].unique()}")


if __name__ == "__main__":
    main()
