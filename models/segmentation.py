"""Segmentation model
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder


def decoder_block(in_channels: int, out_channels: int) -> nn.Module:
    """Standard U-Net decoder block: two 3x3 Convs with BN and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Stage 5
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = decoder_block(1024, 512) # f5 (512) + up5 (512)

        # Stage 4
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = decoder_block(1024, 256) # f4 (512) + up4 (512)

        # Stage 3
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = decoder_block(512, 128)  # f3 (256) + up3 (256)

        # Stage 2
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = decoder_block(256, 64)   # f2 (128) + up2 (128)

        # Stage 1
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = decoder_block(128, 64)   # f1 (64) + up1 (64)

        # Final Classifier
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # Encoder (Contracting Path)
        bottleneck, features = self.encoder(x, return_features=True)

        # Decoder (Expansive Path with Skip Connections)
        # Up5
        d5 = self.up5(bottleneck)
        d5 = torch.cat([features["f5"], d5], dim=1)
        d5 = self.dec5(d5)

        # Up4
        d4 = self.up4(d5)
        d4 = torch.cat([features["f4"], d4], dim=1)
        d4 = self.dec4(d4)

        # Up3
        d3 = self.up3(d4)
        d3 = torch.cat([features["f3"], d3], dim=1)
        d3 = self.dec3(d3)

        # Up2
        d2 = self.up2(d3)
        d2 = torch.cat([features["f2"], d2], dim=1)
        d2 = self.dec2(d2)

        # Up1
        d1 = self.up1(d2)
        d1 = torch.cat([features["f1"], d1], dim=1)
        d1 = self.dec1(d1)

        # Final Output
        return self.final_conv(d1)
