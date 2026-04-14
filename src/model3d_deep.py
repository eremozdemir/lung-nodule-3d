"""
Deep3DCNN — wide and deep 3D CNN for genuine volumetric CT datasets.

Designed for NoduleMNIST3D and LUNA16, where the input is a genuine
28×28×28 voxel patch with real 3D spatial structure.  The architecture is
~4× larger than Small3DCNN to extract richer feature hierarchies.

Architecture:
    Stem  :  1 → 32 ch,  3×3×3 conv + BN + ReLU
    Stage 1: ResBlock3D(32→64)   + MaxPool  → (B, 64,  14³)
    Stage 2: ResBlock3D(64→128)  + MaxPool  → (B, 128,  7³)
    Stage 3: ResBlock3D(128→256) + MaxPool  → (B, 256,  3³)
    Stage 4: ResBlock3D(256,256)            → (B, 256,  3³)  no pool
    GAP  →  Dropout(p)  →  FC(256→1)

Parameter count: ~3.53 M  (~4× Small3DCNN)
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


class Deep3DCNN(nn.Module):
    """
    Wide and deep 3D CNN (~3.53 M params).
    Suitable for genuine 3D volumetric patches (NoduleMNIST3D, LUNA16).

    Key differences vs Small3DCNN:
      - Doubled channel widths at every stage (16/32/64/128 → 32/64/128/256)
      - Extra MaxPool between stages 2 and 3 (28→14→7→3 spatial resolution)
      - Higher default dropout (0.5 vs 0.3)
    """

    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.res1 = ResBlock3D(32,  64)   # pool → (B, 64,  14³)
        self.res2 = ResBlock3D(64,  128)  # pool → (B, 128,  7³)
        self.res3 = ResBlock3D(128, 256)  # pool → (B, 256,  3³)
        self.res4 = ResBlock3D(256, 256)  # no pool → (B, 256, 3³)
        self.gap  = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.drop = nn.Dropout(dropout_p)
        self.fc   = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(self.stem(x))   # (B,  32, 14³)
        x = self.pool(self.res1(x))   # (B,  64,  7³)
        x = self.pool(self.res2(x))   # (B, 128,  3³)
        x = self.res3(x)              # (B, 256,  3³)
        x = self.res4(x)              # (B, 256,  3³)
        x = self.gap(x).flatten(1)   # (B, 256)
        x = self.drop(x)
        return self.fc(x).squeeze(1) # (B,)

    def get_feature_maps(self, x):
        maps = {}
        x = self.pool(self.stem(x));  maps["stem_pool"] = x.detach().cpu()
        x = self.pool(self.res1(x));  maps["res1_pool"] = x.detach().cpu()
        x = self.pool(self.res2(x));  maps["res2_pool"] = x.detach().cpu()
        x = self.res3(x);             maps["res3"]      = x.detach().cpu()
        x = self.res4(x);             maps["res4"]      = x.detach().cpu()
        return maps
