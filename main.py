"""
End-to-end pipeline, run in this order. Each stage writes its outputs to
CFG.output_dir / CFG.manifest_dir so later stages (and you, reviewing the
numbers) don't depend on keeping earlier stages' Python objects in memory.

    python dataset_audit.py          # STAGE 0 — do this first, always
    python main.py --stage kfold     # STAGE 1 — 5-fold CV per backbone
    python main.py --stage ablation  # STAGE 2 — component + TTA-only ablation
    python main.py --stage test      # STAGE 3 — final ensemble on held-out test
    python main.py --stage xai       # STAGE 4 — needs lesion_masks/ populated
    python benchmark_efficiency.py   # STAGE 5 — params/FLOPs/latency (+ ONNX export)

"""
import argparse
import os
import json
import torch
import pandas as pd

from config import CFG
from train import run_kfold
from evaluate import EnsembleModel, evaluate_test_set, significance_tests, fold_level_ensemble_macro_f1
from ablation import run_component_ablation, run_tta_only_ablation
from xai import evaluate_xai_iou


def stage_kfold():
    manifest = pd.read_csv(f"{CFG.manifest_dir}/manifest.csv")
    num_classes = len(CFG.class_names)
    all_fold_results = {}
    best_models = {}
    best_fold_idx = {}  # backbone -> which fold its best_models[backbone] came from.
    # Without this, any downstream evaluation that needs a leakage-free
    # validation split for a specific best model (e.g. TTA-only ablation)
    # has no way to know which fold's val set is actually held-out for that
    # model — evaluating against the wrong fold's val split means testing
    # partly on data the model was trained on, since every image not in
    # fold k's val set is in fold k's train set for k != that fold.

    for backbone in CFG.backbones:
        print(f"\n{'='*60}\nTraining {backbone} — {CFG.n_folds}-fold CV\n{'='*60}")
        df, fold_models = run_kfold(manifest, backbone, num_classes, CFG)
        df.to_csv(f"{CFG.output_dir}/kfold_{backbone}.csv", index=False)
        all_fold_results[backbone] = df
        # keep the fold with highest macro_f1 for the final ensemble
        best_idx = df["macro_f1"].idxmax()
        chosen_fold = int(df.loc[best_idx, "fold"])
        best_models[backbone] = fold_models[chosen_fold]
        best_fold_idx[backbone] = chosen_fold
        print(df[["fold", "accuracy", "macro_f1", "macro_auc"]])

    torch.save(best_models, f"{CFG.checkpoint_dir}/best_models.pt")
    with open(f"{CFG.checkpoint_dir}/best_fold_idx.json", "w") as f:
        json.dump(best_fold_idx, f, indent=2)
    summary = pd.concat(all_fold_results.values())
    summary.to_csv(f"{CFG.output_dir}/kfold_all_backbones_summary.csv", index=False)
    print("\n📊 Per-backbone mean +/- std (accuracy / macro-F1 / macro-AUC):")
    print(summary.groupby("backbone")[["accuracy", "macro_f1", "macro_auc"]].agg(["mean", "std"]))
    return all_fold_results, best_models


def _recover_best_fold_idx(cfg):
    """Get {backbone: fold} for the best model per backbone. Prefers the
    best_fold_idx.json written by stage_kfold() going forward; falls back to
    recomputing from the already-saved kfold_{backbone}.csv files for runs
    that predate that fix (same selection rule: argmax macro_f1 across
    folds) — no retraining needed to recover this."""
    idx_path = f"{cfg.checkpoint_dir}/best_fold_idx.json"
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            return {k: int(v) for k, v in json.load(f).items()}
    recovered = {}
    for backbone in cfg.backbones:
        df = pd.read_csv(f"{cfg.output_dir}/kfold_{backbone}.csv")
        recovered[backbone] = int(df.loc[df["macro_f1"].idxmax(), "fold"])
    print(f"[stage_ablation] best_fold_idx.json not found — recovered fold "
          f"indices from kfold_*.csv instead: {recovered}")
    return recovered


def stage_ablation():
    manifest = pd.read_csv(f"{CFG.manifest_dir}/manifest.csv")
    num_classes = len(CFG.class_names)

    component_df = run_component_ablation(manifest, num_classes, CFG)
    component_df.to_csv(f"{CFG.output_dir}/component_ablation_all_backbones.csv", index=False)
    print(component_df.pivot(index="step", columns="backbone", values="accuracy"))

    best_models = torch.load(f"{CFG.checkpoint_dir}/best_models.pt",
                              weights_only=False)  # trusted, our own file — stores full
                              # CashewNet objects, not just state dicts; PyTorch >=2.6
                              # defaults weights_only=True and refuses to unpickle custom
                              # classes without this
    best_fold_idx = _recover_best_fold_idx(CFG)
    # IMPORTANT: each backbone's best model must be evaluated against ITS OWN
    # held-out fold's val split, not a fixed fold==0. A model whose best fold
    # was e.g. fold 3 would otherwise be tested partly on fold 0's images,
    # which were part of that model's training data — inflating accuracy
    # (this is exactly what produced the ~99.9% "no_tta" numbers instead of
    # the ~98% the component ablation table shows for the same backbones).
    tta_df = run_tta_only_ablation(manifest, num_classes, best_models, best_fold_idx, CFG)
    tta_df.to_csv(f"{CFG.output_dir}/tta_only_ablation.csv", index=False)
    print(tta_df)


def stage_test():
    manifest = pd.read_csv(f"{CFG.manifest_dir}/manifest.csv")
    num_classes = len(CFG.class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_models = torch.load(f"{CFG.checkpoint_dir}/best_models.pt",
                              weights_only=False)  # trusted, our own file — stores full
                              # CashewNet objects, not just state dicts; PyTorch >=2.6
                              # defaults weights_only=True and refuses to unpickle custom
                              # classes without this

    ensemble = EnsembleModel(list(best_models.values()), device)
    result = evaluate_test_set(ensemble, manifest, CFG.class_names, CFG, device)

    print(f"Test accuracy: {result['accuracy']*100:.2f}% "
          f"(95% CI {result['accuracy_95ci'][0]*100:.2f}-{result['accuracy_95ci'][1]*100:.2f}%)")
    print(f"Macro F1: {result['macro_f1']:.4f} "
          f"(95% CI {result['macro_f1_95ci'][0]:.4f}-{result['macro_f1_95ci'][1]:.4f})")
    print(f"Macro AUC: {result['macro_auc']:.4f}")

    # Per-backbone vs. ensemble comparison on the SAME held-out test set
    # Each single backbone is
    # wrapped as a one-model EnsembleModel so evaluate_test_set's TTA
    # handling applies identically to every row — an apples-to-apples
    # comparison, and consistent with how the 3-model ensemble above was
    # evaluated (both get TTA), so the delta reflects ensembling's own
    # contribution rather than a TTA-vs-no-TTA confound. No retraining
    # needed — every model here is already a saved checkpoint.
    print("\n📊 Individual backbones vs. ensemble on held-out test set:")
    comparison_rows = []
    for backbone, model in best_models.items():
        single_result = evaluate_test_set(EnsembleModel([model], device), manifest,
                                           CFG.class_names, CFG, device)
        comparison_rows.append({
            "model": backbone,
            "accuracy": single_result["accuracy"],
            "accuracy_95ci_lo": single_result["accuracy_95ci"][0],
            "accuracy_95ci_hi": single_result["accuracy_95ci"][1],
            "macro_precision": single_result["macro_precision"],
            "macro_recall": single_result["macro_recall"],
            "macro_f1": single_result["macro_f1"],
            "macro_f1_95ci_lo": single_result["macro_f1_95ci"][0],
            "macro_f1_95ci_hi": single_result["macro_f1_95ci"][1],
            "macro_auc": single_result["macro_auc"],
        })
    comparison_rows.append({
        "model": "CashewNet (3-model ensemble)",
        "accuracy": result["accuracy"],
        "accuracy_95ci_lo": result["accuracy_95ci"][0],
        "accuracy_95ci_hi": result["accuracy_95ci"][1],
        "macro_precision": result["macro_precision"],
        "macro_recall": result["macro_recall"],
        "macro_f1": result["macro_f1"],
        "macro_f1_95ci_lo": result["macro_f1_95ci"][0],
        "macro_f1_95ci_hi": result["macro_f1_95ci"][1],
        "macro_auc": result["macro_auc"],
    })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(f"{CFG.output_dir}/test_set_model_comparison.csv", index=False)
    print(comparison_df.to_string(index=False))

    kfold_summary = pd.read_csv(f"{CFG.output_dir}/kfold_all_backbones_summary.csv")
    baseline_scores = {
        b: kfold_summary[kfold_summary.backbone == b]["macro_f1"].tolist()
        for b in CFG.backbones
    }
    # Build the ensemble's OWN per-fold macro-F1 scores by combining each
    # fold's three backbone checkpoints and evaluating on that fold's own
    # held-out val split (see evaluate.fold_level_ensemble_macro_f1) — this
    # is required for a valid paired comparison. An earlier version of this
    # function substituted one baseline's own fold scores here instead
    ensemble_fold_scores = fold_level_ensemble_macro_f1(manifest, CFG.backbones, num_classes, CFG, device)
    if len(ensemble_fold_scores) == CFG.n_folds and len(baseline_scores) >= 1:
        stats_df = significance_tests(ensemble_fold_scores, baseline_scores)
        stats_df.to_csv(f"{CFG.output_dir}/significance_tests.csv", index=False)
        print(stats_df)
    else:
        print(f"[stage_test] WARNING: got {len(ensemble_fold_scores)} ensemble fold "
              f"scores, expected {CFG.n_folds} — skipping significance tests rather "
              f"than risk a silently mismatched/misleading comparison.")

    with open(f"{CFG.output_dir}/test_results.json", "w") as f:
        json.dump({k: v for k, v in result.items() if k not in ("labels", "preds", "probs")},
                   f, indent=2, default=str)


def stage_xai():
    manifest = pd.read_csv(f"{CFG.manifest_dir}/manifest.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_models = torch.load(f"{CFG.checkpoint_dir}/best_models.pt",
                              weights_only=False)  # trusted, our own file — stores full
                              # CashewNet objects, not just state dicts; PyTorch >=2.6
                              # defaults weights_only=True and refuses to unpickle custom
                              # classes without this
    # XAI is run per-backbone (CAM tooling needs a single conv-based model,
    # not the ensemble as a black box) — use the EfficientNetV2 model as the
    # primary reported backbone, consistent with v1's Fig 15.
    model = best_models[CFG.backbones[0]]
    test_df = manifest[manifest.split == "test"]
    df, summary = evaluate_xai_iou(model, test_df, CFG.class_names, CFG.mask_dir, device)
    df.to_csv(f"{CFG.output_dir}/xai_iou_raw.csv", index=False)
    summary.to_csv(f"{CFG.output_dir}/xai_iou_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["kfold", "ablation", "test", "xai"], required=True)
    args = parser.parse_args()

    CFG.ensure_dirs()
    if not os.path.exists(f"{CFG.manifest_dir}/manifest.csv"):
        raise SystemExit("Run `python dataset_audit.py` first — no manifest.csv found.")

    {"kfold": stage_kfold, "ablation": stage_ablation,
     "test": stage_test, "xai": stage_xai}[args.stage]()
