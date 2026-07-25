"""Loss functions (Focal+LS and plain CE for ablation) and EMA weight averaging."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, num_classes, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=self.num_classes).float()
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + self.smoothing / self.num_classes
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        ce_loss = -(targets * log_probs).sum(dim=1)
        p_t = (targets * probs).sum(dim=1)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()


class PlainCrossEntropy(nn.Module):
    """Used for the 'CrossEntropy' ablation row — must support
    soft (mixup/cutmix) targets the same way FocalLoss does, so the comparison
    isolates the loss function, not the label format."""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        if targets.dim() == 1:
            return F.cross_entropy(inputs, targets)
        log_probs = F.log_softmax(inputs, dim=1)
        return -(targets * log_probs).sum(dim=1).mean()


def build_loss(loss_type, num_classes, gamma=2.0, smoothing=0.1):
    if loss_type == "focal_ls":
        return FocalLossWithLabelSmoothing(num_classes, gamma, smoothing)
    if loss_type == "cross_entropy":
        return PlainCrossEntropy(num_classes)
    raise ValueError(loss_type)


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = (1.0 - self.decay) * p.detach() + self.decay * self.shadow[n]

    @torch.no_grad()
    def apply_shadow(self):
        self.backup = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.copy_(self.backup[n])
        self.backup = {}
