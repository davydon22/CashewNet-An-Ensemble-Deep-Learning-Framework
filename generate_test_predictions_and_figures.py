"""
Reruns ensemble inference ONCE on the held-out test set and derives every
downstream artifact that needs raw per-image predictions — confusion
matrix, per-class report, calibration (reliability diagram + Brier score),
and a sample-predictions grid — since stage_test() in main.py deliberately
excluded labels/preds/probs from its saved JSON, and all of these need that
same underlying data. Inference-only: no training, no gradient computation.

Run from the project root, inside the container:
    python3 generate_test_predictions_and_figures.py
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, brier_score_loss
from sklearn.calibration import calibration_curve
from torch.utils.data import DataLoader
from PIL import Image

from config import CFG
from datasets import ManifestDataset, MultiScaleTransform
from evaluate import EnsembleModel

OUT_DIR = CFG.output_dir
os.makedirs(OUT_DIR, exist_ok=True)


def run_inference():
    """Single pass over the test set, returns a DataFrame with one row per
    image: path, true class, predicted class, confidence, and the full
    per-class probability vector — the reusable artifact everything else in
    this script (and future analyses) derives from."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(f"{CFG.manifest_dir}/manifest.csv")
    test_df = manifest[manifest.split == "test"].reset_index(drop=True)

    best_models = torch.load(f"{CFG.checkpoint_dir}/best_models.pt", weights_only=False)
    ensemble = EnsembleModel(list(best_models.values()), device)

    ds = ManifestDataset(test_df, MultiScaleTransform(CFG.img_size, is_train=False))
    loader = DataLoader(ds, batch_size=CFG.batch_size, shuffle=False, num_workers=4)

    rows = []
    idx = 0
    for images, labels in loader:
        probs = ensemble.predict_batch(images, use_tta=True).cpu().numpy()
        preds = probs.argmax(axis=1)
        for i in range(len(labels)):
            rows.append({
                "path": test_df.loc[idx, "path"],
                "true_label": int(labels[i]),
                "true_class": CFG.class_names[int(labels[i])],
                "pred_label": int(preds[i]),
                "pred_class": CFG.class_names[int(preds[i])],
                "confidence": float(probs[i, preds[i]]),
                "probs": probs[i].tolist(),
            })
            idx += 1

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/test_predictions_full.csv", index=False)
    print(f"Saved {len(df)} test predictions to test_predictions_full.csv")
    return df


def build_confusion_matrix(df):
    y_true, y_pred = df["true_label"], df["pred_label"]
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    pd.DataFrame(cm, index=CFG.class_names, columns=CFG.class_names).to_csv(f"{OUT_DIR}/confusion_matrix_counts.csv")
    pd.DataFrame(cm_norm, index=CFG.class_names, columns=CFG.class_names).to_csv(f"{OUT_DIR}/confusion_matrix_normalized.csv")

    report = classification_report(y_true, y_pred, target_names=CFG.class_names, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(f"{OUT_DIR}/per_class_report.csv")

    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CFG.class_names))); ax.set_xticklabels(CFG.class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(CFG.class_names))); ax.set_yticklabels(CFG.class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (ensemble, test set, row-normalized)", fontsize=10)
    for i in range(len(CFG.class_names)):
        for j in range(len(CFG.class_names)):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n(n={cm[i,j]})", ha="center", va="center",
                     fontsize=8, color="white" if cm_norm[i, j] > 0.5 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=220, facecolor="white")
    plt.close()
    print("Saved confusion_matrix.png, confusion_matrix_counts.csv, confusion_matrix_normalized.csv, per_class_report.csv")


def build_calibration_analysis(df):
    correct = (df["true_label"] == df["pred_label"]).astype(int).values
    confidence = df["confidence"].values

    prob_true, prob_pred = calibration_curve(correct, confidence, n_bins=10, strategy="uniform")
    brier = brier_score_loss(correct, confidence)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(prob_pred, prob_true, marker="o", color="#6FA8DC")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_xlabel("Mean predicted confidence"); axes[0].set_ylabel("Observed accuracy")
    axes[0].set_title(f"Reliability diagram (Brier score = {brier:.4f})", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].hist(confidence[correct == 1], bins=25, alpha=0.6, label="Correct", color="#93C47D", density=True)
    axes[1].hist(confidence[correct == 0], bins=25, alpha=0.6, label="Incorrect", color="#E06666", density=True)
    axes[1].set_xlabel("Confidence"); axes[1].set_ylabel("Density")
    axes[1].set_title("Confidence distribution by correctness", fontsize=10)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/calibration_analysis.png", dpi=220, facecolor="white")
    plt.close()

    with open(f"{OUT_DIR}/calibration_summary.json", "w") as f:
        json.dump({"brier_score": brier, "n_correct": int(correct.sum()), "n_total": len(correct)}, f, indent=2)
    print(f"Saved calibration_analysis.png, calibration_summary.json (Brier score = {brier:.4f})")


def build_sample_predictions_grid(df, n_correct=6, n_incorrect=2, seed=42):
    rng = np.random.RandomState(seed)
    correct_df = df[df.true_label == df.pred_label]
    incorrect_df = df[df.true_label != df.pred_label]

    n_incorrect = min(n_incorrect, len(incorrect_df))
    if n_incorrect < 2:
        print(f"WARNING: only {len(incorrect_df)} misclassified test images exist — "
              f"sample size for qualitative failure-case discussion is small; "
              f"caveat this in the write-up per the earlier statistical-power discussion.")

    picked_correct = correct_df.sample(n=min(n_correct, len(correct_df)), random_state=seed)
    picked_incorrect = incorrect_df.sample(n=n_incorrect, random_state=seed) if n_incorrect > 0 else incorrect_df
    picked = pd.concat([picked_correct, picked_incorrect])

    n = len(picked)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes, picked.iterrows()):
        img = Image.open(row["path"]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        color = "green" if row["true_label"] == row["pred_label"] else "red"
        ax.set_title(f"True: {row['true_class']}\nPred: {row['pred_class']} ({row['confidence']:.2f})",
                      fontsize=9, color=color)
    for ax in axes[n:]:
        ax.axis("off")

    plt.suptitle("Sample ensemble predictions with confidence (test set)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/sample_predictions.png", dpi=200, facecolor="white")
    plt.close()
    print(f"Saved sample_predictions.png ({len(picked_correct)} correct, {n_incorrect} incorrect)")


if __name__ == "__main__":
    df = run_inference()
    build_confusion_matrix(df)
    build_calibration_analysis(df)
    build_sample_predictions_grid(df)
    print("\nDone. New files in cashewnet_outputs/: "
          "test_predictions_full.csv, confusion_matrix.png, confusion_matrix_counts.csv, "
          "confusion_matrix_normalized.csv, per_class_report.csv, calibration_analysis.png, "
          "calibration_summary.json, sample_predictions.png")
