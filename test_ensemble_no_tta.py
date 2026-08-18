"""
decouples ensembling's contribution from TTA's contribution with a
full 2x2 design:
    (1) single backbone, no TTA
    (2) single backbone, with TTA
    (3) ensemble, no TTA 
    (4) ensemble, with TTA

Only cell (3) requires new inference: evaluate the 3-backbone ensemble using
only the identity view per backbone (no flips/rotations averaged in),
instead of the existing 6-view TTA scheme. Reuses the existing trained
checkpoints -- no retraining.

Run from the project root, inside the container:
    python3 test_ensemble_no_tta.py
"""
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CFG
from datasets import ManifestDataset, MultiScaleTransform
from evaluate import EnsembleModel

OUT_DIR = CFG.output_dir


def evaluate_ensemble(ensemble, df, device, use_tta):
    transform = MultiScaleTransform(CFG.img_size, is_train=False)
    ds = ManifestDataset(df, transform)
    loader = DataLoader(ds, batch_size=CFG.batch_size, shuffle=False, num_workers=4)

    correct, total = 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            probs = ensemble.predict_batch(images, use_tta=use_tta)
            preds = probs.argmax(dim=1).cpu()
            correct += (preds == labels).sum().item()
            total += len(labels)
            all_probs.append(probs.cpu())
            all_labels.append(labels)
    accuracy = correct / total
    probs_cat = torch.cat(all_probs)
    labels_cat = torch.cat(all_labels)
    # macro F1 via simple confusion-matrix-free per-class computation
    preds_cat = probs_cat.argmax(dim=1)
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
    ensemble = EnsembleModel(list(best_models.values()), device)

    print("Evaluating ensemble WITHOUT TTA (identity view only, 3 backbones combined)...")
    acc_no_tta, f1_no_tta = evaluate_ensemble(ensemble, test_df, device, use_tta=False)
    print(f"  Ensemble, no TTA:   accuracy={acc_no_tta*100:.2f}%  macro_f1={f1_no_tta:.4f}")

    print("Evaluating ensemble WITH TTA (6-view, 3 backbones combined) -- should match the "
          "existing headline result (99.32%) as a sanity check...")
    acc_with_tta, f1_with_tta = evaluate_ensemble(ensemble, test_df, device, use_tta=True)
    print(f"  Ensemble, with TTA: accuracy={acc_with_tta*100:.2f}%  macro_f1={f1_with_tta:.4f}")

    if abs(acc_with_tta * 100 - 99.32) > 0.05:
        print(f"\nWARNING: the 'with TTA' sanity-check accuracy ({acc_with_tta*100:.2f}%) does not "
              f"match the previously reported 99.32% headline result closely -- investigate before "
              f"trusting the 'no TTA' comparison (could indicate a different checkpoint set, a "
              f"different test split, or a bug in this script).")

    result = pd.DataFrame([
        {"configuration": "Ensemble, no TTA", "accuracy_pct": acc_no_tta * 100, "macro_f1": f1_no_tta},
        {"configuration": "Ensemble, with TTA", "accuracy_pct": acc_with_tta * 100, "macro_f1": f1_with_tta},
    ])
    result["delta_from_no_tta_pp"] = (result["accuracy_pct"] - acc_no_tta * 100)
    result.to_csv(f"{OUT_DIR}/ensemble_tta_decoupling.csv", index=False)
    print(f"\nSaved {OUT_DIR}/ensemble_tta_decoupling.csv")
    print("\nCombine this with the existing (single-backbone no-TTA/with-TTA) to complete "
          "the full 2x2 comparison requested: single-backbone-no-TTA, single-backbone-with-TTA "
          "(already reported), ensemble-no-TTA (this script), ensemble-with-TTA (already reported, "
          "used as a sanity-check cross-reference above).")


if __name__ == "__main__":
    main()
