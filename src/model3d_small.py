"""
Small3DCNN — lightweight 3D CNN for pseudo-3D (2D-slice-stacked) datasets.

Designed for IQ-OTH:NCCD and LungcancerDataSet, where each "volume" is a
single 2D CT slice repeated 28 times along the depth axis.  Because there
is no genuine depth variation, a shallow/light architecture is sufficient and
avoids overfitting on the easy visual patterns in these datasets.

Architecture:
    Stem  :  1 → 16 ch,  3×3×3 conv + BN + ReLU
    Stage 1: ResBlock3D(16→32)  + MaxPool  → (B, 32, 14³)
    Stage 2: ResBlock3D(32→64)  + MaxPool  → (B, 64,  7³)
    Stage 3: ResBlock3D(64,128)            → (B, 128, 7³)
    GAP  →  Dropout(p)  →  FC(128→1)

Parameter count: ~884 K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    """3D residual block: two 3×3×3 convs with a skip connection."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.skip  = (
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_ch),
            )
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class Small3DCNN(nn.Module):
    """
    Lightweight 3D CNN (~884 K params).
    Suitable for pseudo-3D volumes (same 2D slice stacked 28×).
    """

    def __init__(self, dropout_p: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.res1 = ResBlock3D(16, 32)   # → (B, 32, 14³)
        self.res2 = ResBlock3D(32, 64)   # → (B, 64,  7³)
        self.res3 = ResBlock3D(64, 128)  # → (B,128,  7³)  no pool
        self.gap  = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.drop = nn.Dropout(dropout_p)
        self.fc   = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(self.stem(x))   # (B, 16, 14³)
        x = self.pool(self.res1(x))   # (B, 32,  7³)
        x = self.res2(x)              # (B, 64,  7³)
        x = self.res3(x)              # (B,128,  7³)
        x = self.gap(x).flatten(1)   # (B, 128)
        x = self.drop(x)
        return self.fc(x).squeeze(1) # (B,)

    def get_feature_maps(self, x):
        maps = {}
        x = self.pool(self.stem(x));  maps["stem_pool"] = x.detach().cpu()
        x = self.pool(self.res1(x));  maps["res1_pool"] = x.detach().cpu()
        x = self.res2(x);             maps["res2"]      = x.detach().cpu()
        x = self.res3(x);             maps["res3"]      = x.detach().cpu()
        return maps
