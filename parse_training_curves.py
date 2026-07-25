"""
Parses logs_kfold.txt to reconstruct per-epoch training curves, without
needing to rerun anything — every real training run already printed this
data to the log, it just wasn't saved in structured form.

IMPORTANT CAVEAT this script handles explicitly: your log file accumulated
across multiple separate container sessions (crashes, host reboots,
resumes), each launched with `tee -a` (append). This means:
  - A fold that was interrupted mid-training BEFORE its checkpoint was
    saved will appear as an incomplete epoch sequence, followed by a fresh
    "Epoch 1/50" restart for that same fold in a later session.
  - A fold that was resumed via the [resume] checkpoint-loading logic
    (train.py's resume_from_checkpoint) has NO epoch-by-epoch curve at
    all — training was skipped entirely, only a single summary line was
    printed.

This script's strategy: group epoch lines into "segments" (a new segment
starts whenever the epoch number resets to 1), tag each segment with the
backbone it belongs to, and — WITHIN each backbone — keep only the LAST N
segments, where N = cfg.n_folds (5), on the assumption that the final
(most recent) attempt at each fold position is the one whose checkpoint
was actually saved and used. This is a heuristic, not a guarantee: verify
the printed "segments found per backbone" count against your own knowledge
of how many times each backbone's training was interrupted, and visually
sanity-check the resulting figure (e.g., every fold's curve should
generally trend upward/plateau, not show a sudden drop mid-curve that
would indicate two different attempts got concatenated).

Run from the project root:
    python3 parse_training_curves.py logs_kfold.txt
"""
import sys
import re
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HEADER_RE = re.compile(r"Training (\S+)\s*[—_-]+\s*\d+-fold CV")
EPOCH_RE = re.compile(
    r"Epoch (\d+)/(\d+) \| train_loss=([\d.]+) train_acc=([\d.]+)% \| "
    r"val_acc=([\d.]+)% val_f1=([\d.]+) val_auc=(nan|[\d.]+)"
)
RESUME_RE = re.compile(r"\[resume\] (\S+) fold(\d+): loaded checkpoint from epoch (\S+), val_acc=([\d.]+)%")


def parse(log_path, n_folds=5):
    current_backbone = None
    segments = []          # list of dicts: {backbone, rows: [...]}
    current_segment = None
    resumed_folds = []     # list of (backbone, fold, val_acc) with no curve data

    with open(log_path) as f:
        for line in f:
            m = HEADER_RE.search(line)
            if m:
                current_backbone = m.group(1)
                continue

            m = RESUME_RE.search(line)
            if m:
                resumed_folds.append({"backbone": m.group(1), "fold": int(m.group(2)),
                                       "checkpoint_epoch": m.group(3), "val_acc": float(m.group(4))})
                continue

            m = EPOCH_RE.search(line)
            if m and current_backbone:
                epoch = int(m.group(1))
                row = {
                    "backbone": current_backbone, "epoch": epoch,
                    "train_loss": float(m.group(3)), "train_acc": float(m.group(4)),
                    "val_acc": float(m.group(5)), "val_f1": float(m.group(6)),
                    "val_auc": None if m.group(7) == "nan" else float(m.group(7)),
                }
                if epoch == 1:
                    if current_segment is not None:
                        segments.append(current_segment)
                    current_segment = {"backbone": current_backbone, "rows": [row]}
                elif current_segment is not None:
                    current_segment["rows"].append(row)
    if current_segment is not None:
        segments.append(current_segment)

    # Keep only the LAST n_folds segments per backbone (see module docstring
    # for the reasoning/caveat behind this heuristic).
    by_backbone = {}
    for seg in segments:
        by_backbone.setdefault(seg["backbone"], []).append(seg)

    kept = {}
    for backbone, segs in by_backbone.items():
        print(f"{backbone}: {len(segs)} training segment(s) found in the log "
              f"(keeping the last {min(n_folds, len(segs))})")
        kept[backbone] = segs[-n_folds:]

    return kept, resumed_folds


def to_dataframe(kept):
    rows = []
    for backbone, segs in kept.items():
        for fold_idx, seg in enumerate(segs):
            for r in seg["rows"]:
                rows.append({**r, "fold": fold_idx})
    return pd.DataFrame(rows)


def plot(df, resumed_folds, out_path="cashewnet_outputs/training_curves.png"):
    backbones = df["backbone"].unique()
    fig, axes = plt.subplots(2, len(backbones), figsize=(5.5 * len(backbones), 8), squeeze=False)

    for col, backbone in enumerate(backbones):
        sub = df[df.backbone == backbone]
        for fold in sorted(sub.fold.unique()):
            fold_df = sub[sub.fold == fold].sort_values("epoch")
            axes[0, col].plot(fold_df.epoch, fold_df.train_loss, "--", alpha=0.5, linewidth=1)
            axes[0, col].plot(fold_df.epoch, fold_df.val_f1, alpha=0.8, linewidth=1.3, label=f"fold {fold} (val F1)")
            axes[1, col].plot(fold_df.epoch, fold_df.train_acc, "--", alpha=0.5, linewidth=1)
            axes[1, col].plot(fold_df.epoch, fold_df.val_acc, alpha=0.8, linewidth=1.3, label=f"fold {fold}")
        axes[0, col].set_title(f"{backbone}\n(dashed=train loss, solid=val macro-F1)", fontsize=9)
        axes[0, col].set_xlabel("Epoch"); axes[0, col].grid(alpha=0.3)
        axes[1, col].set_title("(dashed=train acc, solid=val acc)", fontsize=9)
        axes[1, col].set_xlabel("Epoch"); axes[1, col].set_ylabel("Accuracy (%)"); axes[1, col].grid(alpha=0.3)
        axes[1, col].legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor="white")
    print(f"Saved {out_path}")

    if resumed_folds:
        print(f"\n{len(resumed_folds)} fold(s) were resumed from checkpoint and have NO epoch curve "
              f"(training was skipped, only final val_acc is known):")
        for r in resumed_folds:
            print(f"  {r['backbone']} fold {r['fold']}: val_acc={r['val_acc']:.2f}% "
                  f"(from checkpoint saved at epoch {r['checkpoint_epoch']})")


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "logs_kfold.txt"
    kept, resumed_folds = parse(log_path)
    df = to_dataframe(kept)
    df.to_csv("cashewnet_outputs/training_curves_raw.csv", index=False)
    print(f"\nSaved cashewnet_outputs/training_curves_raw.csv ({len(df)} epoch records)")
    plot(df, resumed_folds)
