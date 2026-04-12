import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    """
    Simple 3D residual block: two conv layers with a skip connection.
    If in_ch != out_ch, a 1x1 conv is used to match dimensions.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)

        # 1x1 projection to match channel dims for the skip connection
        self.skip = (
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.skip(x))   # residual addition
        return out


class Small3DCNN(nn.Module):
    """
    Small 3D CNN for 28x28x28 nodule cubes.
    Output is a single logit for binary classification.
    Uses residual blocks for improved feature extraction.
    """
    def __init__(self, dropout_p: float = 0.3):
        super().__init__()

        # Stem: first conv to get to 16 channels
        self.stem = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Residual stages
        self.res1 = ResBlock3D(16, 32)   # after pool: (B, 32, 14, 14, 14)
        self.res2 = ResBlock3D(32, 64)   # after pool: (B, 64, 7, 7, 7)
        self.res3 = ResBlock3D(64, 128)  # no pool: (B, 128, 7, 7, 7)

        self.gap  = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.drop = nn.Dropout(dropout_p)
        self.fc   = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, 1, D, H, W) where D=H=W=28
        x = self.pool(self.stem(x))   # (B, 16, 14, 14, 14)
        x = self.pool(self.res1(x))   # (B, 32, 7, 7, 7)
        x = self.res2(x)              # (B, 64, 7, 7, 7)
        x = self.res3(x)              # (B, 128, 7, 7, 7)
        x = self.gap(x).flatten(1)   # (B, 128)
        x = self.drop(x)
        x = self.fc(x).squeeze(1)    # (B,)
        return x

    def get_feature_maps(self, x):
        """
        Returns intermediate feature maps for visualization.
        Returns a dict of layer_name -> feature map tensor (kept on CPU).
        """
        maps = {}
        x = self.pool(self.stem(x));  maps["stem_pool"] = x.detach().cpu()
        x = self.pool(self.res1(x));  maps["res1_pool"] = x.detach().cpu()
        x = self.res2(x);             maps["res2"] = x.detach().cpu()
        x = self.res3(x);             maps["res3"] = x.detach().cpu()
        return maps


class Deep3DCNN(nn.Module):
    """
    Deep and wider 3D CNN for 28×28×28 nodule cubes (Trial 4).

    Architectural changes vs Small3DCNN:
      - Stem: 16 channels → 32 channels
      - Residual stage widths doubled: (32→64) → (64→128) → (128→256)
      - Third MaxPool added: spatial resolution 28 → 14 → 7 → 3
      - Dropout: 0.3 → 0.4 (more regularisation for the larger capacity)
      - Final feature vector: 128 → 256

    Parameter count: ~3.5 M (vs ~882 K for Small3DCNN, ~4× larger).
    """
    def __init__(self, dropout_p: float = 0.4):
        super().__init__()

        # Stem: 1 → 32 channels
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Three residual stages, all followed by pooling
        self.res1 = ResBlock3D(32, 64)     # pool → (B, 64,  14, 14, 14)
        self.res2 = ResBlock3D(64, 128)    # pool → (B, 128,  7,  7,  7)
        self.res3 = ResBlock3D(128, 256)   # pool → (B, 256,  3,  3,  3)

        self.gap  = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.drop = nn.Dropout(dropout_p)
        self.fc   = nn.Linear(256, 1)

    def forward(self, x):
        # x: (B, 1, D, H, W) where D=H=W=28
        x = self.pool(self.stem(x))   # (B,  32, 14, 14, 14)
        x = self.pool(self.res1(x))   # (B,  64,  7,  7,  7)
        x = self.pool(self.res2(x))   # (B, 128,  3,  3,  3)
        x = self.res3(x)              # (B, 256,  3,  3,  3)
        x = self.gap(x).flatten(1)   # (B, 256)
        x = self.drop(x)
        x = self.fc(x).squeeze(1)    # (B,)
        return x

    def get_feature_maps(self, x):
        """
        Returns intermediate feature maps for visualization.
        Returns a dict of layer_name -> feature map tensor (kept on CPU).
        """
        maps = {}
        x = self.pool(self.stem(x));  maps["stem_pool"] = x.detach().cpu()
        x = self.pool(self.res1(x));  maps["res1_pool"] = x.detach().cpu()
        x = self.pool(self.res2(x));  maps["res2_pool"] = x.detach().cpu()
        x = self.res3(x);             maps["res3"]      = x.detach().cpu()
        return maps