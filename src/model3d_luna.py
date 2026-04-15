"""
LUNA3DCNN -- dedicated 3D CNN for LUNA16 nodule-presence detection.

Task
----
Binary classification of 32x32x32 CT patches:
    Positive (1): patch centred on a radiologist-annotated pulmonary nodule
    Negative (0): randomly sampled background tissue patch (no nodule)

This is fundamentally different from NoduleMNIST3D malignancy classification:
  - Features are macroscopic: a dense spherical structure vs uniform lung parenchyma
  - Class separation is visually clear even at 32x32x32 resolution
  - A compact, focused model with appropriate regularisation generalises well

Design choices
--------------
  - 3 residual stages + MaxPool each: 32 -> 16 -> 8 -> 4 spatial resolution
  - Channel widths: 32 -> 64 -> 128 -> 256 (same family as Deep3DCNN)
  - No cutout augmentation (risk of erasing the nodule centre)
  - Lower dropout (0.3 vs 0.5): the task is simpler, less regularisation needed
  - AdaptiveAvgPool3d handles any input spatial size

Architecture
------------
    Stem  : 1 -> 32 ch,  3x3x3 conv + BN + ReLU
    Stage 1: ResBlock3D(32->64)   + MaxPool  -> (B,  64,  16^3)
    Stage 2: ResBlock3D(64->128)  + MaxPool  -> (B, 128,   8^3)
    Stage 3: ResBlock3D(128->256) + MaxPool  -> (B, 256,   4^3)
    GAP  ->  Dropout(0.3)  ->  FC(256->1)

Parameter count: ~2.0 M
Input:  (B, 1, D, H, W)  normalised CT patch, any spatial size
Output: (B,)  raw logit for BCEWithLogitsLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    """3D residual block: two 3x3x3 convs with a skip connection."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class LUNA3DCNN(nn.Module):
    """
    Compact 3D CNN for LUNA16 nodule-presence detection (~2.0 M params).

    Trained and evaluated exclusively on LUNA16.  Unlike Deep3DCNN, which must
    learn subtle malignancy texture features, this model only needs to distinguish
    a dense spherical nodule from uniform background lung parenchyma -- a
    macroscopic visual difference well-suited to a 3-stage residual network.

    Parameters
    ----------
    dropout_p : float
        Dropout probability before the classification head (default 0.3).
    """

    def __init__(self, dropout_p: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.res1 = ResBlock3D(32,  64)    # pool -> (B,  64, 16^3 from 32^3)
        self.res2 = ResBlock3D(64,  128)   # pool -> (B, 128,  8^3)
        self.res3 = ResBlock3D(128, 256)   # pool -> (B, 256,  4^3)
        self.gap  = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.drop = nn.Dropout(dropout_p)
        self.fc   = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, D, H, W)  normalised CT patch

        Returns
        -------
        logits : (B,)
        """
        x = self.pool(self.stem(x))   # (B,  32, 16^3)
        x = self.pool(self.res1(x))   # (B,  64,  8^3)
        x = self.pool(self.res2(x))   # (B, 128,  4^3)
        x = self.res3(x)              # (B, 256,  4^3)
        x = self.gap(x).flatten(1)   # (B, 256)
        x = self.drop(x)
        return self.fc(x).squeeze(1)  # (B,)

    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """
        Return intermediate feature maps for visualization.
        Maps are detached and moved to CPU.
        """
        maps = {}
        x = self.pool(self.stem(x));  maps["stem_pool"] = x.detach().cpu()
        x = self.pool(self.res1(x));  maps["res1_pool"] = x.detach().cpu()
        x = self.pool(self.res2(x));  maps["res2_pool"] = x.detach().cpu()
        x = self.res3(x);             maps["res3"]      = x.detach().cpu()
        return maps
