# CashewNet Experiment

This is a full CashewNet codebase. It is meant to be run on your
own GPU workstatio. Treat everything below as ready-to-execute.

## files

| File | Fixes |
|---|---|---|
| `dataset_audit.py` | Perceptual-hash near-duplicate clustering + group-aware split (`StratifiedGroupKFold`) so no duplicate image can appear in both train and test; background/occlusion proxy stats per class; per-class-per-split counts |
| `datasets.py` | Manifest-driven (not live directory scan) so the leakage-safe split is enforced everywhere downstream; documents soft-label Mixup/CutMix behaviour in prose-ready form | 
| `models.py` | `use_eca` / `use_fusion` toggles are now real, buildable model variants | 
| `losses.py` | Adds `PlainCrossEntropy` (soft-label aware) so the "CrossEntropy" ablation row is genuinely comparable | 
| `train.py` | 5-fold default (was 3); every fold logs accuracy and macro-F1 and macro-AUC; model selection by macro-F1 | 
| `ablation.py` | Component ablation run on all three backbones, not just EfficientNetV2; adds a TTA-only (no ensemble) row to isolate TTA's contribution | 
| `evaluate.py` | Wilcoxon signed-rank alongside paired t-test; bootstrap 95% CIs on test accuracy/F1; ensemble+TTA evaluated once on a held-out test split that k-fold training never touches |
| `xai.py` | GradCAM++, Score-CAM, Eigen-CAM side by side; IoU against human-annotated lesion masks, reported **per class** | R1-#12, R2-#6, R3-#13 |
| `benchmark_efficiency.py` | Params/FLOPs/latency/GPU-mem measured consistently across backbones + lightweight baselines (MobileNetV3, ShuffleNetV2, SwinV2-Tiny); ONNX export for real edge-device benchmarking | 
| `config.py` | Single source of truth for every hyperparameter/toggle, so can audit exact settings in one place |


## How to run

```bash
pip install -r requirements.txt --break-system-packages

# STAGE 0 — mandatory first step. Point raw_dataset_root in config.py at your
# unsplit image pool (all classes in subfolders, no pre-existing train/val/test
# split). This clusters near-duplicates and writes a leakage-safe manifest.
python dataset_audit.py

# STAGE 1 — 5-fold CV per backbone, writes checkpoints + per-fold metrics
python main.py --stage kfold

# STAGE 2 — component ablation (all backbones) + TTA-only ablation
python main.py --stage ablation

# STAGE 3 — final ensemble+TTA evaluation on the held-out test split,
# with bootstrap CIs and significance tests
python main.py --stage test

# STAGE 4 — quantitative XAI (needs lesion_masks/ populated first)
python main.py --stage xai

# STAGE 5 — efficiency benchmarking + ONNX export for edge deployment
python benchmark_efficiency.py
```

Every stage writes CSV/JSON to `cashewnet_outputs/` — those files are your new
Tables 1, 2, 5, 6, 7, 8 and the new XAI-IoU table, generated from a single
consistent pipeline instead of the scattered one-off cells in the original
notebook.

## First thing to check once Stage 0 finishes

`dataset_audit.py` will print (and hard-fail on, via `leakage_check`) how many
near-duplicate image clusters exist in your pool. **Read that number before
doing anything else.
