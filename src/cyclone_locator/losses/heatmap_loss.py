import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapMSE(nn.Module):
    """MSE sulla heatmap; può restituire la media per campione (reduction='none')."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, reduction: str = "mean"):
        # pred, target: (B,1,H,W)
        if pred.shape != target.shape:
            raise ValueError("pred/target shape mismatch")
        if reduction == "none":
            # ritorna (B,) media sui pixel per ciascun sample
            return F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1)
        return F.mse_loss(pred, target, reduction=reduction)
