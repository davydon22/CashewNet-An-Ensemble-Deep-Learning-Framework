"""
Manifest-driven datasets.

datasets are built from the audited manifest.csv
(image_path,label,group,split,fold) produced by dataset_audit.py, never from
a live directory scan. This guarantees every consumer of the data respects
the leakage-safe split — it is impossible to accidentally re-shuffle across
duplicate-image groups downstream.
"""
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T


class MultiScaleTransform:
    def __init__(self, size=224, is_train=True):
        self.size = size
        self.is_train = is_train
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image):
        if self.is_train:
            scale = random.uniform(0.8, 1.2)
            new_size = int(self.size * scale)
            image = TF.resize(image, (new_size, new_size))
            image = TF.center_crop(image, self.size)
            if random.random() > 0.5:
                image = TF.hflip(image)
            if random.random() > 0.5:
                image = TF.rotate(image, random.uniform(-15, 15))
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
        else:
            image = TF.resize(image, (256, 256))
            image = TF.center_crop(image, self.size)

        image = TF.to_tensor(image)
        image = self.normalize(image)
        return image


class ManifestDataset(Dataset):
    """Plain dataset: hard integer labels. Used for validation/test and for
    ablation variants that don't use Mixup/CutMix."""

    def __init__(self, manifest_df, transform=None):
        self.paths = manifest_df["path"].tolist()
        self.labels = manifest_df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


class MixupCutmixManifestDataset(Dataset):
    """Same augmentation logic MixupCutmixDataset, but manifest-driven
    and with the multiclass soft-label behaviour documented in prose here: 
    a mixed sample's target is `lam * one_hot(y1) + (1-lam) * one_hot(y2)`,
    i.e. the *training signal itself* reflects that the pixels come from two
    classes, rather than assigning either label a full weight of 1."""

    def __init__(self, manifest_df, transform, num_classes,
                 mixup_alpha=0.8, cutmix_alpha=1.0, prob=0.5):
        self.paths = manifest_df["path"].tolist()
        self.labels = manifest_df["label"].tolist()
        self.transform = transform
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.length = len(self.paths)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return self.length

    def one_hot(self, label):
        vec = torch.zeros(self.num_classes)
        vec[label] = 1.0
        return vec

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        label = self.labels[idx]

        if random.random() < self.prob:
            if random.random() > 0.5:
                return self.mixup(image, label)
            return self.cutmix(image, label)

        if self.transform:
            image = self.transform(image)
        return image, self.one_hot(label)

    def _load_second(self):
        idx2 = random.randint(0, self.length - 1)
        image2 = Image.open(self.paths[idx2]).convert("RGB")
        return image2, self.labels[idx2]

    def mixup(self, image, label):
        image2, label2 = self._load_second()
        image = self.transform(image)
        _, h, w = image.shape
        image2 = TF.resize(image2, (h, w))
        image2 = (TF.to_tensor(image2) - self.mean) / self.std
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        mixed = lam * image + (1 - lam) * image2
        mixed_label = lam * self.one_hot(label) + (1 - lam) * self.one_hot(label2)
        return mixed, mixed_label

    def cutmix(self, image, label):
        image2, label2 = self._load_second()
        image = self.transform(image)
        _, h, w = image.shape
        image2 = TF.resize(image2, (h, w))
        image2 = (TF.to_tensor(image2) - self.mean) / self.std
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        cut_w, cut_h = int(w * np.sqrt(1 - lam)), int(h * np.sqrt(1 - lam))
        cx, cy = np.random.randint(w), np.random.randint(h)
        x1, x2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)
        y1, y2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)
        image[:, y1:y2, x1:x2] = image2[:, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        mixed_label = lam * self.one_hot(label) + (1 - lam) * self.one_hot(label2)
        return image, mixed_label
