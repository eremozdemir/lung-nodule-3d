import torch
import torch.nn as nn
import torch.nn.functional as F


class Small3DCNN(nn.Module):
    """
    Small 3D CNN for 28x28x28 nodule cubes.
    Output is a single logit for binary classification.
    Adds one extra conv stage and dropout before the classifier head.
    """
    def __init__(self, dropout_p: float = 0.3):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)

        # Extra capacity: 64 -> 128
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.drop = nn.Dropout(dropout_p)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, 1, D, H, W) where D=H=W=28
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (B, 16, 14, 14, 14)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (B, 32, 7, 7, 7)
        x = F.relu(self.bn3(self.conv3(x)))             # (B, 64, 7, 7, 7)
        x = F.relu(self.bn4(self.conv4(x)))             # (B, 128, 7, 7, 7)
        x = self.gap(x).flatten(1)                      # (B, 128)
        x = self.drop(x)
        x = self.fc(x).squeeze(1)                       # (B,)
        return x
    
    
    