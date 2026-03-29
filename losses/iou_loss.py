"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Invalid reduction type: {reduction}")
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""

        p_xc, p_yc, p_w, p_h = pred_boxes.unbind(dim=-1)
        t_xc, t_yc, t_w, t_h = target_boxes.unbind(dim=-1)

        # Convert to (x1, y1, x2, y2)
        p_x1 = p_xc - p_w / 2.0
        p_y1 = p_yc - p_h / 2.0
        p_x2 = p_xc + p_w / 2.0
        p_y2 = p_yc + p_h / 2.0

        t_x1 = t_xc - t_w / 2.0
        t_y1 = t_yc - t_h / 2.0
        t_x2 = t_xc + t_w / 2.0
        t_y2 = t_yc + t_h / 2.0

        # Intersection coordinates
        inter_x1 = torch.max(p_x1, t_x1)
        inter_y1 = torch.max(p_y1, t_y1)
        inter_x2 = torch.min(p_x2, t_x2)
        inter_y2 = torch.min(p_y2, t_y2)

        # Intersection area
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h

        # Union area
        pred_area = p_w * p_h
        target_area = t_w * t_h
        union_area = pred_area + target_area - inter_area

        # Compute IoU
        iou = inter_area / (union_area + self.eps)

        # Loss
        loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss