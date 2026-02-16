import math
from typing import Tuple

import torch
from torch import nn

from cyclone_locator.utils.dsnt import spatial_softmax_2d


def compute_energy_features(
    heatmap_logits: torch.Tensor,
    dsnt_tau: float,
    topk: int | None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      E: (B,1) top-K mean(sigmoid(logits))
      C: (B,1) concentration = 1 - H(P)/log(H*W)
    """
    prob = spatial_softmax_2d(heatmap_logits, tau=float(dsnt_tau))
    if prob.ndim == 4 and prob.shape[1] == 1:
        p = prob.squeeze(1)
    else:
        p = prob
    b, h, w = p.shape
    entropy = -(p * (p + eps).log()).flatten(1).sum(dim=1)
    denom = math.log(max(1, h * w))
    conc = 1.0 - entropy / denom
    conc = conc.clamp(0.0, 1.0).unsqueeze(1)

    logits = heatmap_logits.squeeze(1).flatten(1)
    k = int(topk or logits.shape[1])
    k = min(max(1, k), logits.shape[1])
    vals, _ = torch.topk(logits, k=k, dim=1)
    energy = torch.sigmoid(vals).mean(dim=1, keepdim=True)
    return energy, conc


class EnergyFusion(nn.Module):
    """
    z_tot = b0 + wE*E + wC*C + wH*z_head_logit
    """

    def __init__(self, b0: float = 0.0, wE: float = 1.0, wC: float = 1.0, wH: float = 1.0) -> None:
        super().__init__()
        self.b0 = nn.Parameter(torch.tensor(float(b0)))
        self.wE = nn.Parameter(torch.tensor(float(wE)))
        self.wC = nn.Parameter(torch.tensor(float(wC)))
        self.wH = nn.Parameter(torch.tensor(float(wH)))

    def forward(self, energy: torch.Tensor, conc: torch.Tensor, head_logit: torch.Tensor) -> torch.Tensor:
        return self.b0 + self.wE * energy + self.wC * conc + self.wH * head_logit
