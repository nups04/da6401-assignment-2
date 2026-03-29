"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer implementing inverted dropout.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0 <= p < 1):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        - Training: Apply binary mask and scale by 1/(1-p).
        - Inference: Return input as-is.
        """
        if not self.training or self.p == 0:
            return x
        
        # Generate binary mask (1 with probability 1-p)
        mask = (torch.rand_like(x) > self.p).float()
        
        # Inverted dropout scaling
        return (x * mask) / (1.0 - self.p)
