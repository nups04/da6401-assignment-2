"""Localization modules
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Regression head for [x_center, y_center, width, height]
        # Coordinates are normalized between [0, 1], so we use Sigmoid
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 4),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format.
        """
        x = self.encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.regressor(x)
