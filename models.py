"""
CashewNet architecture, with real toggles for ablation.

"""
import numpy as np
import torch
import torch.nn as nn
import timm


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((np.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1
        k_size = max(k_size, 3)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c).unsqueeze(1)
        y = self.sigmoid(self.conv(y))
        y = y.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        return x * y.expand_as(x)


class Identity2d(nn.Module):
    """No-op used in place of ECABlock when use_eca=False, so the rest of the
    forward pass doesn't need branching logic."""
    def forward(self, x):
        return x


class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels=512, use_eca=True):
        super().__init__()
        self.num_scales = len(in_channels_list)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
            for c in in_channels_list
        ])
        self.attentions = nn.ModuleList([
            ECABlock(out_channels) if use_eca else Identity2d()
            for _ in in_channels_list
        ])
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * self.num_scales, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, features):
        import torch.nn.functional as F
        target_size = features[0].shape[2:]
        aligned = []
        for i, feat in enumerate(features):
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            feat = self.convs[i](feat)
            feat = self.attentions[i](feat)
            aligned.append(feat)
        fused = torch.cat(aligned, dim=1)
        fused = self.fusion_conv(fused)
        fused = self.global_pool(fused)
        return torch.flatten(fused, 1)


class SingleScalePool(nn.Module):
    """Used when use_fusion=False (ablation): just GAP the last backbone stage
    and project to the same 512-d width, so the classification head is
    unchanged across ablation variants — isolating the fusion module's effect."""
    def __init__(self, in_channels, out_channels=512):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, features):
        x = self.proj(features[-1])
        x = self.pool(x)
        return torch.flatten(x, 1)


class CashewNet(nn.Module):
    """
    Args exposed for ablation:
      use_eca:    apply ECA channel attention per backbone stage
      use_fusion: fuse the last 3 stages (True) vs. use only the last stage (False)
    """

    def __init__(self, model_name, num_classes, dropout_rate=0.3,
                 stochastic_depth_rate=0.1, use_eca=True, use_fusion=True,
                 pretrained=True):
        super().__init__()
        self.use_eca = use_eca
        self.use_fusion = use_fusion

        # pretrained=False is useful for offline smoke-testing architecture
        # shapes without needing to hit the HF Hub / timm weight servers.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True,
            drop_path_rate=stochastic_depth_rate
        )
        self.feature_dims = self.backbone.feature_info.channels()

        self.stage_eca = nn.ModuleList([
            ECABlock(d) if use_eca else Identity2d() for d in self.feature_dims
        ])

        if use_fusion:
            fusion_dims = self.feature_dims[-3:]
            self.fusion = MultiScaleFeatureFusion(fusion_dims, out_channels=512, use_eca=False)
            # NOTE: use_eca is already applied per-stage above via self.stage_eca;
            # the fusion module's *internal* ECA is kept off to avoid double-counting
            # attention when isolating the fusion ablation from the ECA ablation.
        else:
            self.fusion = SingleScalePool(self.feature_dims[-1], out_channels=512)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(dropout_rate), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        processed = []
        for i, feat in enumerate(features):
            if feat.dim() == 4 and feat.shape[1] != self.feature_dims[i]:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            feat = self.stage_eca[i](feat)
            processed.append(feat)

        if self.use_fusion:
            fused = self.fusion(processed[-3:])
        else:
            fused = self.fusion(processed)

        return self.classifier(fused)


def build_model(backbone_name, num_classes, use_eca=True, use_fusion=True, pretrained=True):
    return CashewNet(backbone_name, num_classes, use_eca=use_eca, use_fusion=use_fusion,
                      pretrained=pretrained)
