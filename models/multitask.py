"""Unified multi-task model
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3):
        super().__init__()
        
        # 1. Shared Backbone
        self.encoder = VGG11Encoder(in_channels=in_channels)
        
        # 2. Classification Head components
        cls_model = VGG11Classifier(num_classes=num_breeds, in_channels=in_channels)
        self.avgpool = cls_model.avgpool
        self.classifier = cls_model.classifier
        
        # 3. Localization Head components
        loc_model = VGG11Localizer(in_channels=in_channels)
        self.regressor = loc_model.regressor
        
        # 4. Segmentation Decoder Head components
        seg_model = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)
        self.up5 = seg_model.up5
        self.dec5 = seg_model.dec5
        self.up4 = seg_model.up4
        self.dec4 = seg_model.dec4
        self.up3 = seg_model.up3
        self.dec3 = seg_model.dec3
        self.up2 = seg_model.up2
        self.dec2 = seg_model.dec2
        self.up1 = seg_model.up1
        self.dec1 = seg_model.dec1
        self.final_conv = seg_model.final_conv

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        # --- Shared Backbone ---
        bottleneck, features = self.encoder(x, return_features=True)
        
        # --- Common Pooling for Classification and Localization ---
        pooled = self.avgpool(bottleneck)
        flat = torch.flatten(pooled, 1)
        
        # --- 1. Classification Head ---
        cls_out = self.classifier(flat)
        
        # --- 2. Localization Head ---
        loc_out = self.regressor(flat)
        
        # --- 3. Segmentation Head (Symmetric Decoder) ---
        d5 = self.up5(bottleneck)
        d5 = torch.cat([features["f5"], d5], dim=1)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat([features["f4"], d4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([features["f3"], d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([features["f2"], d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([features["f1"], d1], dim=1)
        d1 = self.dec1(d1)

        seg_out = self.final_conv(d1)
        
        return {
            'classification': cls_out,
            'localization': loc_out,
            'segmentation': seg_out
        }
