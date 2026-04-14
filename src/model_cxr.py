"""
CXRClassifier — fine-tuned ResNet-18 for chest X-ray binary cancer classification.

Motivation
----------
Chest X-rays are genuine 2D images.  The pseudo-3D stacking approach used for
the earlier Small3DCNN (repeating the same 2D slice 28× along a depth axis)
wastes capacity and cannot extract any information from the depth direction.

A pretrained 2D ResNet-18 is a far stronger inductive prior:
  - ImageNet weights already encode edge detectors, texture filters, and shape
    detectors that transfer effectively to radiograph classification.
  - 224×224 input preserves considerably more spatial detail than the 28×28
    downsampling used in the pseudo-3D pipeline.
  - Fine-tuning requires only ~5K samples to converge — ideal for the CXR
    training set size.

Architecture
------------
  - ResNet-18 backbone (pretrained on ImageNet via torchvision)
  - First Conv2d adapted: 3-channel → 1-channel grayscale by averaging the
    RGB filter weights across channels — this preserves the learned filter
    energy and avoids random re-initialisation.
  - Global Average Pool (built into ResNet) → (B, 512)
  - Dropout(p) → Linear(512 → 1)
  - Output: raw logit for BCEWithLogitsLoss

Input:   (B, 1, 224, 224)  float32, normalised with mean=0.5, std=0.5
Output:  (B,)  raw logit

Parameter count: ~11.18 M (ResNet-18 backbone) + 513 (head)
"""

import torch
import torch.nn as nn


class CXRClassifier(nn.Module):
    """
    ResNet-18 backbone fine-tuned for 1-channel chest X-ray binary classification.

    Parameters
    ----------
    dropout_p : float
        Dropout probability applied before the final linear layer (default 0.5).
    pretrained : bool
        If True, initialise from ImageNet weights.  Set False only for testing.
    """

    def __init__(self, dropout_p: float = 0.5, pretrained: bool = True):
        super().__init__()
        import torchvision.models as tv

        weights = tv.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv.resnet18(weights=weights)

        # ── Adapt first conv: 3-channel RGB → 1-channel grayscale ─────────────
        # Average the three RGB filter maps so that grayscale luminance values
        # produce the same activation magnitudes as the original RGB inputs.
        orig_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        if pretrained:
            backbone.conv1.weight.data = orig_conv.weight.data.mean(
                dim=1, keepdim=True
            )

        # ── Feature extractor: ResNet up to (and including) adaptive avg-pool ─
        # children() order: conv1, bn1, relu, maxpool, layer1..4, avgpool, fc
        # We drop the final FC (index -1) and keep everything else.
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 512, 1, 1)

        self.drop = nn.Dropout(dropout_p)
        self.fc   = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, H, W)  normalised grayscale CXR tensor

        Returns
        -------
        logits : (B,)
        """
        x = self.features(x).flatten(1)   # (B, 512)
        return self.fc(self.drop(x)).squeeze(1)  # (B,)
