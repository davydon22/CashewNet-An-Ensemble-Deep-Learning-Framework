"""
Training loop + k-fold driver.

  - NUM_FOLDS 5 by default, read from config.
  - Every fold logs accuracy, macro-F1, AND macro-AUC 
  - `train_one_config` takes an explicit (backbone, use_eca, use_fusion,
    use_mixup, use_ema, loss_type) tuple, so the SAME function trains both the
    main model and every ablation variant — no more hand-built one-off models
    that silently diverge from what's used in the final ensemble.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from tqdm import tqdm

from config import CFG
from datasets import ManifestDataset, MixupCutmixManifestDataset, MultiScaleTransform
from models import build_model
from losses import build_loss, EMA


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self, model, device, train_loader, val_loader, criterion, optimizer,
                 scheduler=None, use_ema=True, ema_decay=0.999, use_amp=True, grad_clip=1.0,
                 num_classes=4):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None
        self.grad_clip = grad_clip
        self.num_classes = num_classes
        self.use_ema = use_ema
        if use_ema:
            self.ema = EMA(self.model, decay=ema_decay)

    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, targets in tqdm(self.train_loader, desc="train", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            if self.use_amp:
                with autocast(device_type='cuda'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            if self.use_ema:
                self.ema.update()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            true = targets.argmax(dim=1) if targets.dim() == 2 else targets
            correct += (preds == true).sum().item()
            total += true.size(0)
        if self.scheduler is not None:
            self.scheduler.step()
        return total_loss / len(self.train_loader), 100 * correct / total

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        if self.use_ema:
            self.ema.apply_shadow()

        total_loss, all_preds, all_labels, all_probs = 0, [], [], []
        for inputs, targets in tqdm(self.val_loader, desc="val", leave=False):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with autocast(device_type='cuda') if self.use_amp else torch.enable_grad():
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets)
            total_loss += loss.item()
            # Cast to fp32 before softmax: under autocast, outputs (and a naive
            # softmax(outputs)) are fp16, and fp16 rounding means probability
            # rows don't sum to *exactly* 1.0. sklearn's roc_auc_score has a
            # strict check for this in the multiclass OvR case and raises
            # ValueError when it's off even slightly — which the broad
            # `except ValueError` below was silently turning into nan on
            # every single epoch, regardless of fold or backbone. This was
            # confirmed by reproducing the exact ValueError with fp16 probs
            # in isolation, and by ruling out a missing-class explanation
            # against the real manifest (every fold has all 4 classes).
            probs = F.softmax(outputs.float(), dim=1)
            preds = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        if self.use_ema:
            self.ema.restore()

        acc = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        try:
            macro_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        except ValueError:
            macro_auc = float("nan")

        return {
            "loss": total_loss / len(self.val_loader),
            "accuracy": 100 * acc,
            "macro_f1": macro_f1,
            "macro_auc": macro_auc,
            "labels": all_labels, "preds": all_preds, "probs": all_probs,
        }

    def fit(self, epochs, patience, checkpoint_path=None):
        best_f1, best_epoch, best_state, no_improve = -1, 0, None, 0
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
                   "val_macro_f1": [], "val_macro_auc": []}

        for epoch in range(epochs):
            tr_loss, tr_acc = self.train_epoch()
            val = self.validate()
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val["loss"])
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(val["accuracy"])
            history["val_macro_f1"].append(val["macro_f1"])
            history["val_macro_auc"].append(val["macro_auc"])

            print(f"Epoch {epoch+1}/{epochs} | train_loss={tr_loss:.4f} train_acc={tr_acc:.2f}% "
                  f"| val_acc={val['accuracy']:.2f}% val_f1={val['macro_f1']:.4f} "
                  f"val_auc={val['macro_auc']:.4f}")

            # Model selection on macro-F1, not raw accuracy — more appropriate
            # for a 4-class problem where per-class balance matters.
            if val["macro_f1"] > best_f1:
                best_f1, best_epoch, no_improve = val["macro_f1"], epoch + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                if checkpoint_path:
                    torch.save({"epoch": best_epoch, "model_state_dict": best_state,
                                "best_macro_f1": best_f1}, checkpoint_path)
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        final_val = self.validate()
        return {"model": self.model, "best_epoch": best_epoch, "best_macro_f1": best_f1,
                "final_val": final_val, "history": history}


def train_one_config(manifest_df, fold, backbone, num_classes, cfg=CFG,
                      use_eca=True, use_fusion=True, use_mixup=True, use_ema=True,
                      loss_type="focal_ls", device=None, tag=""):
    """Train one (backbone x ablation-toggle) config on one fold. Returns the
    Trainer.fit() result dict plus config metadata, ready to be logged into a
    single results table shared by the main experiment and the ablation study."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed + fold)

    train_df = manifest_df[(manifest_df.fold == fold) & (manifest_df.split == "train")]
    val_df = manifest_df[(manifest_df.fold == fold) & (manifest_df.split == "val")]

    train_tf = MultiScaleTransform(cfg.img_size, is_train=True)
    val_tf = MultiScaleTransform(cfg.img_size, is_train=False)

    if use_mixup:
        train_ds = MixupCutmixManifestDataset(train_df, train_tf, num_classes,
                                               cfg.mixup_alpha, cfg.cutmix_alpha, cfg.mix_prob)
    else:
        train_ds = ManifestDataset(train_df, train_tf)
    val_ds = ManifestDataset(val_df, val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = build_model(backbone, num_classes, use_eca=use_eca, use_fusion=use_fusion)
    criterion = build_loss(loss_type, num_classes, cfg.focal_gamma, cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    trainer = Trainer(model, device, train_loader, val_loader, criterion, optimizer, scheduler,
                       use_ema=use_ema, ema_decay=cfg.ema_decay, use_amp=cfg.use_amp,
                       grad_clip=cfg.grad_clip, num_classes=num_classes)

    ckpt_name = f"{tag or backbone}_fold{fold}.pth"
    ckpt_path = os.path.join(cfg.checkpoint_dir, ckpt_name)
    result = trainer.fit(cfg.epochs, cfg.patience, checkpoint_path=ckpt_path)

    result.update({"backbone": backbone, "fold": fold, "use_eca": use_eca, "use_fusion": use_fusion,
                    "use_mixup": use_mixup, "use_ema": use_ema, "loss_type": loss_type, "tag": tag})
    return result


def resume_from_checkpoint(manifest_df, fold, backbone, num_classes, ckpt_path, cfg=CFG,
                            use_eca=True, use_fusion=True, device=None):
    """Reload an already-completed fold's best weights and re-run validation
    to reconstruct the exact same result dict train_one_config would have
    returned — without retraining. Safe because validate() has no dropout or
    augmentation active (model.eval() + deterministic val transform), so the
    reconstructed metrics match what was originally logged before the
    interruption, to within floating-point reproducibility."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    val_df = manifest_df[(manifest_df.fold == fold) & (manifest_df.split == "val")]
    val_tf = MultiScaleTransform(cfg.img_size, is_train=False)
    val_ds = ManifestDataset(val_df, val_tf)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = build_model(backbone, num_classes, use_eca=use_eca, use_fusion=use_fusion)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # Minimal Trainer instance purely to reuse its .validate() method — no
    # optimizer/criterion/EMA needed since we're not training.
    trainer = Trainer(model, device, train_loader=None, val_loader=val_loader,
                       criterion=None, optimizer=None, use_ema=False,
                       use_amp=cfg.use_amp, num_classes=num_classes)
    final_val = trainer.validate()
    print(f"[resume] {backbone} fold{fold}: loaded checkpoint from epoch "
          f"{ckpt.get('epoch', '?')}, val_acc={final_val['accuracy']:.2f}% "
          f"val_f1={final_val['macro_f1']:.4f} (skipped retraining)")
    return {"model": model, "best_epoch": ckpt.get("epoch", 0),
            "best_macro_f1": ckpt.get("best_macro_f1", final_val["macro_f1"]),
            "final_val": final_val}


def run_kfold(manifest_df, backbone, num_classes, cfg=CFG, resume=True, **ablation_kwargs):
    """Train across all cfg.n_folds and return a results DataFrame with
    accuracy / macro-F1 / macro-AUC per fold.

    resume=True (default): if checkpoints/{backbone}_fold{N}.pth already
    exists on disk, that fold is not retrained — its result is reconstructed
    from the saved weights instead. This matters because a container/host
    restart mid-run used to mean losing all progress back to fold 0; now only the fold that was
    actively training when the interruption hit needs to redo work.
    Ablation-variant runs pass resume=False, since their checkpoint filenames
    are shared across ablation steps by design (see ablation.py) and must
    not be confused with the main run's checkpoints.
    """
    rows = []
    fold_models = []
    for fold in range(cfg.n_folds):
        ckpt_name = f"{backbone}_fold{fold}.pth"
        ckpt_path = os.path.join(cfg.checkpoint_dir, ckpt_name)

        if resume and os.path.exists(ckpt_path):
            res = resume_from_checkpoint(manifest_df, fold, backbone, num_classes,
                                          ckpt_path, cfg,
                                          use_eca=ablation_kwargs.get("use_eca", True),
                                          use_fusion=ablation_kwargs.get("use_fusion", True))
        else:
            res = train_one_config(manifest_df, fold, backbone, num_classes, cfg, **ablation_kwargs)

        fv = res["final_val"]
        rows.append({
            "backbone": backbone, "fold": fold, "accuracy": fv["accuracy"],
            "macro_f1": fv["macro_f1"], "macro_auc": fv["macro_auc"],
            "best_epoch": res["best_epoch"], **{k: v for k, v in ablation_kwargs.items()},
        })
        fold_models.append(res["model"])
    df = pd.DataFrame(rows)
    return df, fold_models
