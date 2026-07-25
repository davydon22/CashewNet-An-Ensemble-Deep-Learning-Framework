"""
CashewNet — central configuration.

Every path, hyperparameter, and toggle used across the pipeline lives here
"""
from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class Config:
    # ---------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------
    raw_dataset_root: str = "cashew_leaf_dataset_raw"      # unsplit pool, all classes in subfolders
    manifest_dir: str = "manifests"                         # audited / group-split CSVs written here
    output_dir: str = "cashewnet_outputs"
    mask_dir: str = "lesion_masks"                           # binary lesion masks for IoU-XAI,
    checkpoint_dir: str = "checkpoints"

    # ---------------------------------------------------------------
    # Classes — keep in sync with raw_dataset_root subfolder names
    # ---------------------------------------------------------------
    class_names: List[str] = field(default_factory=lambda: [
        "anthracnose", "healthy", "leaf_miner", "red_rust"
    ])

    # ---------------------------------------------------------------
    # Dataset audit (near-duplicate / leakage check)
    # ---------------------------------------------------------------
    phash_hamming_threshold: int = 6   # <=6 bits differ => treated as near-duplicate cluster
    test_fraction: float = 0.15        # held out ONCE, group-aware, never touched until final eval
    n_folds: int = 5                   # folds 5

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------
    img_size: int = 224
    batch_size: int = 16
    epochs: int = 50
    patience: int = 7
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_epochs: int = 5
    seed: int = 42

    # Augmentation
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.5

    # Loss
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1

    # EMA
    ema_decay: float = 0.999

    use_amp: bool = True

    # ---------------------------------------------------------------
    # Backbones
    # ---------------------------------------------------------------
    backbones: List[str] = field(default_factory=lambda: [
        "tf_efficientnetv2_s", "convnext_tiny", "swin_tiny_patch4_window7_224"
    ])
    # Additional lightweight baselines for (edge-focused comparison).
    # NOTE: ShuffleNetV2 does not exist anywhere in timm's model registry
    # (verified via timm.list_models('*shufflenet*') — zero matches), and
    # build_model() always wraps backbones through timm's features_only API,
    # so "shufflenet_v2_x1_0" would hard-fail inside timm.create_model and
    # silently vanish from the efficiency table (caught by benchmark_all's
    # try/except, printed as a warning easy to miss in a long log). Properly
    # wiring a torchvision-only architecture through timm's multi-scale
    # feature-map API is nontrivial and unverified without live testing, 
    lightweight_baselines: List[str] = field(default_factory=lambda: [
        "mobilenetv3_large_100", "mobilenetv2_100", "swinv2_tiny_window8_256"
    ])
    # Additional recent baselines for (classification-native only —
    # RT-DETR / DINO are detection / self-supervised pretraining methods and
    # are not classification-native;)
    modern_baselines: List[str] = field(default_factory=lambda: [
        "efficientvit_b0", "yolo11n-cls"   # yolo11n-cls loaded via ultralytics, not timm
    ])

    # TTA views used at inference (identity + flips + 90/180/270 rotations)
    tta_views: int = 6

    def ensure_dirs(self):
        for d in [self.manifest_dir, self.output_dir, self.checkpoint_dir]:
            os.makedirs(d, exist_ok=True)


CFG = Config()
