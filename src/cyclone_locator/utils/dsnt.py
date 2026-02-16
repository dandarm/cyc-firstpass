import torch


def spatial_softmax_2d(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    logits: (B,1,H,W) or (B,H,W)
    Returns: probabilities with same shape as input (no channel dim added/removed).
    """
    if tau <= 0:
        raise ValueError("tau must be > 0")
    orig_shape = logits.shape
    if logits.ndim == 4 and orig_shape[1] == 1:
        x = logits.squeeze(1)
        squeeze_ch = True
    elif logits.ndim == 3:
        x = logits
        squeeze_ch = False
    else:
        raise ValueError(f"Expected logits shape (B,1,H,W) or (B,H,W), got {orig_shape}")

    b, h, w = x.shape
    x = (x / float(tau)).reshape(b, h * w)
    x = x - x.amax(dim=1, keepdim=True)
    p = torch.softmax(x, dim=1).reshape(b, h, w)
    if squeeze_ch:
        return p.unsqueeze(1)
    return p


def dsnt_expectation(prob: torch.Tensor) -> torch.Tensor:
    """
    prob: (B,1,H,W) or (B,H,W), assumed to sum to 1 over HxW.
    Returns: (B,2) with (x,y) in heatmap pixel coordinates [0..W-1], [0..H-1].
    """
    if prob.ndim == 4 and prob.shape[1] == 1:
        p = prob.squeeze(1)
    elif prob.ndim == 3:
        p = prob
    else:
        raise ValueError(f"Expected prob shape (B,1,H,W) or (B,H,W), got {prob.shape}")

    b, h, w = p.shape
    device = p.device
    dtype = p.dtype
    ys = torch.arange(h, device=device, dtype=dtype)
    xs = torch.arange(w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    exp_x = (p * xx).flatten(1).sum(dim=1)
    exp_y = (p * yy).flatten(1).sum(dim=1)
    return torch.stack([exp_x, exp_y], dim=1)


def heatmap_centroid(heatmap: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    heatmap: (B,1,H,W) or (B,H,W), non-negative weights (not necessarily normalized).
    Returns: (B,2) centroid in heatmap pixel coordinates.
    """
    if heatmap.ndim == 4 and heatmap.shape[1] == 1:
        hm = heatmap.squeeze(1)
    elif heatmap.ndim == 3:
        hm = heatmap
    else:
        raise ValueError(f"Expected heatmap shape (B,1,H,W) or (B,H,W), got {heatmap.shape}")

    hm = hm.clamp_min(0)
    denom = hm.flatten(1).sum(dim=1).clamp_min(eps)
    b, h, w = hm.shape
    device = hm.device
    dtype = hm.dtype
    ys = torch.arange(h, device=device, dtype=dtype)
    xs = torch.arange(w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    exp_x = (hm * xx).flatten(1).sum(dim=1) / denom
    exp_y = (hm * yy).flatten(1).sum(dim=1) / denom
    return torch.stack([exp_x, exp_y], dim=1)

