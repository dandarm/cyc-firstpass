import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapMSE(nn.Module):
    """MSE sulla heatmap, con maschera: calcolata solo sui positivi."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        # pred, target: (B,1,H,W) ; mask: (B,) 1=positivo, 0=negativo
        if pred.shape != target.shape:
            raise ValueError("pred/target shape mismatch")
        B = pred.shape[0]
        loss = (pred - target)**2
        # espandi mask a (B,1,1,1)
        m = mask.view(B, 1, 1, 1).to(pred.dtype)
        loss = loss * m
        denom = m.sum() * pred.shape[2] * pred.shape[3]
        if denom < 1:
            return pred.new_tensor(0.)
        return loss.sum() / denom
