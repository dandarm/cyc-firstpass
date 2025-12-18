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


class HeatmapFocal(nn.Module):
    """Focal loss per heatmap in stile CenterNet (logits -> sigmoid interno)."""

    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    def forward(self, logits, target, reduction: str = "mean"):
        if logits.shape != target.shape:
            raise ValueError("pred/target shape mismatch")
        pred = torch.sigmoid(logits)
        pred = pred.clamp(self.eps, 1.0 - self.eps)
        target = target.clamp(0.0, 1.0)

        # Variante "soft": usa target come peso, evita dipendenza da pixel esattamente == 1
        pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, self.alpha) * target
        neg_weight = torch.pow(1.0 - target, self.beta)
        neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, self.alpha) * neg_weight * (1.0 - target)

        loss = pos_loss + neg_loss  # (B,1,H,W)
        if reduction == "none":
            return loss.flatten(1).mean(dim=1)  # (B,)
        if reduction == "sum":
            return loss.sum()
        return loss.mean()
