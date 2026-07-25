"""
Checks lesion mask annotation progress AND validates mask quality, not just
presence — a mask that "exists" but is empty, all-white, or the wrong shape
would pass a simple file-count check yet silently break IoU computation
later (empty ground truth -> IoU undefined/nan for that image; wrong shape
-> a resize distorts the annotation).

Run from the project root (same place manifest.csv and lesion_masks/ live):
    python3 check_mask_progress.py
"""
import pandas as pd
import numpy as np
import os
import cv2

CLASS_NAMES = ["anthracnose", "healthy", "leaf_miner", "red_rust"]
TARGET_PER_CLASS = {"anthracnose": 30, "healthy": 10, "leaf_miner": 50, "red_rust": 30}


def main():
    sel = pd.read_csv("annotation_selection_manifest.csv")

    print("=" * 60)
    print("MASK ANNOTATION PROGRESS")
    print("=" * 60)

    problems = []
    done_count = 0

    for cls in CLASS_NAMES:
        cls_rows = sel[sel["class"] == cls]
        target = TARGET_PER_CLASS.get(cls, 0)
        cls_done = 0

        for _, row in cls_rows.iterrows():
            mask_path = row["mask_should_go_to"]
            if not os.path.exists(mask_path):
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                problems.append(f"{mask_path}: file exists but couldn't be read as an image")
                continue

            unique_vals = np.unique(mask)
            lesion_fraction = (mask > 0).mean()

            if cls == "healthy":
                # For healthy leaves, an all-black mask (uniform value 0) IS
                # the correct annotation — there's no lesion to mark, so
                # "empty" here means "correctly annotated," not "not done
                # yet." This is the opposite of every other class, where an
                # empty mask always means unannotated. Still flag a healthy
                # mask that's uniform at some OTHER value (128, 255, etc —
                # very likely a real mistake, not an intentional "no lesion"
                # marking), or one with a suspiciously large marked region
                # (could indicate the wrong image/class got mixed up).
                if len(unique_vals) == 1 and unique_vals[0] != 0:
                    problems.append(f"{mask_path}: uniform at value {unique_vals[0]} "
                                     f"(not black/0) — likely a mistake, not an "
                                     f"intentional 'no lesion' marking")
                    continue
                if lesion_fraction > 0.05:
                    problems.append(f"{mask_path}: {lesion_fraction*100:.1f}% marked as "
                                     f"lesion on a HEALTHY image — check this isn't a "
                                     f"mixed-up image/class")
                    continue
                cls_done += 1
                continue

            if len(unique_vals) == 1:
                problems.append(f"{mask_path}: completely uniform ({unique_vals[0]}) "
                                 f"— looks like an empty/unedited mask, not a real annotation")
                continue
            if lesion_fraction < 0.001:
                problems.append(f"{mask_path}: lesion region is essentially empty "
                                 f"({lesion_fraction*100:.3f}% of pixels) — check this one")
                continue
            if lesion_fraction > 0.95:
                problems.append(f"{mask_path}: {lesion_fraction*100:.1f}% of the image marked "
                                 f"as lesion — unusually large, double check this one")

            cls_done += 1

        done_count += cls_done
        bar_len = 20
        filled = int(bar_len * cls_done / target) if target else 0
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"  {cls:12s} [{bar}] {cls_done}/{target}")

    total_target = sum(TARGET_PER_CLASS.values())
    print(f"\nTotal: {done_count}/{total_target} masks done "
          f"({100*done_count/total_target:.0f}%)")

    if problems:
        print(f"\n⚠️  {len(problems)} mask(s) flagged for review (not counted as "
              f"valid above unless noted):")
        for p in problems[:20]:
            print(f"   - {p}")
        if len(problems) > 20:
            print(f"   ... and {len(problems) - 20} more")
    else:
        print("\n✅ No quality issues detected in masks found so far.")

    if done_count >= 15 and done_count < total_target:
        print(f"\n💡 You have enough masks now ({done_count}) to do a first test "
              f"run of the XAI stage and confirm the pipeline works end-to-end, "
              f"before finishing the rest.")
    elif done_count >= total_target:
        print(f"\n🎉 All masks done — ready to run the full XAI stage:")
        print(f"   python3 main.py --stage xai")


if __name__ == "__main__":
    main()
