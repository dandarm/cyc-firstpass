import os, argparse, yaml, time, csv, random
from contextlib import nullcontext
import numpy as np
import torch, torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from cyclone_locator.datasets.med_fullbasin import MedFullBasinDataset
from cyclone_locator.models.simplebaseline import SimpleBaseline
from cyclone_locator.models.x3d_backbone import X3DBackbone
from cyclone_locator.losses.heatmap_loss import HeatmapMSE, HeatmapFocal
from cyclone_locator.utils.dsnt import dsnt_expectation, spatial_softmax_2d
from cyclone_locator.utils.distributed import (
    cleanup_distributed,
    get_resources,
    is_main_process,
    reduce_mean,
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yml")
    ap.add_argument("--train_csv")
    ap.add_argument("--val_csv")
    ap.add_argument("--test_csv")
    ap.add_argument("--image_size", type=int)
    ap.add_argument("--heatmap_stride", type=int)
    ap.add_argument("--heatmap_sigma_px", type=float)
    ap.add_argument("--heatmap_loss", choices=["mse", "focal", "dsnt"], help="Heatmap loss type")
    ap.add_argument("--dsnt_tau", type=float, help="Temperature tau for DSNT softmax2D")
    ap.add_argument("--dsnt_coord_loss", choices=["l1", "l2"], help="Coordinate loss for DSNT")
    ap.add_argument("--peak_pool", choices=["max", "logsumexp"], help="Pooling for presence_from_peak (default: max)")
    ap.add_argument("--peak_tau", type=float, help="Temperature tau for peak_pool=logsumexp")
    ap.add_argument("--w_heatmap_focal_reg", type=float, help="Extra focal heatmap regularizer weight (adds to DSNT+BCE)")
    ap.add_argument("--heatmap_focal_reg_neg_multiplier", type=float, help="Negative multiplier for focal heatmap regularizer (default 0=positives only)")
    ap.add_argument("--backbone")
    ap.add_argument("--temporal_T", type=int)
    ap.add_argument("--temporal_stride", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--bs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--log_dir")
    ap.add_argument("--best_ckpt_start_epoch", type=int)
    ap.add_argument("--grad_accum_steps", type=int, help="Gradient accumulation steps (effective batch = bs * steps)")
    ap.add_argument("--num_workers", type=int, help="Override dataloader workers (use 0 to debug NCCL stalls)")
    ap.add_argument("--dataloader_timeout_s", type=int, help="Timeout (s) for DataLoader get_batch to avoid deadlocks")
    ap.add_argument("--persistent_workers", type=int, choices=[0,1], help="Set persistent_workers (1=keep workers alive between epochs)")
    ap.add_argument("--heatmap_neg_multiplier", type=float, help="Scale factor for heatmap loss on negative samples")
    ap.add_argument("--heatmap_pos_multiplier", type=float, help="Scale factor for heatmap loss on positive samples")
    ap.add_argument("--presence_from_peak", type=int, choices=[0,1], help="If 1, disable presence head and use heatmap peak as presence prob")
    return ap.parse_args()

def set_seed(sd):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sd)


def seed_worker(worker_id):
    """Ensure dataloader workers get different seeds."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def bce_logits(pred, target):
    return nn.functional.binary_cross_entropy_with_logits(pred, target)

def smooth_targets(target, eps: float):
    if eps <= 0.0:
        return target
    return target * (1.0 - eps) + (1.0 - target) * eps

def combine_losses(L_hm, L_pr, loss_cfg):
    hm_w = loss_cfg["w_heatmap"]
    pr_w = loss_cfg["w_presence"]
    return hm_w, pr_w, hm_w * L_hm + pr_w * L_pr


def focal_loss_with_logits(logits, targets, gamma: float = 2.0, alpha: float = None):
    """
    Binary focal loss on logits. Targets are probabilities (after smoothing).
    """
    ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_factor = (1 - p_t).pow(gamma)
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        ce = ce * alpha_t
    return (focal_factor * ce).mean()


def coord_loss_per_sample(pred_xy: torch.Tensor, target_xy: torch.Tensor, loss_type: str) -> torch.Tensor:
    """
    pred_xy, target_xy: (B,2) in heatmap pixel coordinates.
    Returns: (B,) per-sample loss.
    """
    if pred_xy.shape != target_xy.shape or pred_xy.ndim != 2 or pred_xy.shape[1] != 2:
        raise ValueError(f"Expected pred_xy/target_xy shape (B,2), got {pred_xy.shape} vs {target_xy.shape}")
    loss_type = str(loss_type).lower().strip()
    if loss_type in {"l1", "mae"}:
        return (pred_xy - target_xy).abs().sum(dim=1)
    if loss_type in {"l2", "mse"}:
        return ((pred_xy - target_xy) ** 2).sum(dim=1)
    raise ValueError(f"Unknown dsnt_coord_loss: {loss_type}")


def spatial_peak_pool(logits: torch.Tensor, pool: str = "max", tau: float = 1.0) -> torch.Tensor:
    """
    logits: (B,1,H,W)
    Returns: (B,1) pooled logits.
    """
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"Expected logits shape (B,1,H,W), got {logits.shape}")
    pool = str(pool).lower().strip()
    if pool == "max":
        return logits.amax(dim=[-1, -2])
    if pool == "logsumexp":
        if tau <= 0:
            raise ValueError("tau must be > 0 for logsumexp pooling")
        # Do pooling in float32 for stability under AMP / small tau.
        x = (logits / float(tau)).float()
        pooled = torch.logsumexp(x, dim=[-1, -2])
        return float(tau) * pooled
    raise ValueError(f"Unknown peak_pool: {pool}")


def center_mae_px(pred_xy_hm: torch.Tensor, target_xy_hm: torch.Tensor, valid: torch.Tensor, stride: int) -> torch.Tensor:
    """
    pred_xy_hm, target_xy_hm: (B,2) in heatmap pixel coordinates.
    valid: (B,) float/bool mask (1=valid target).
    Returns: scalar MAE (mean over valid samples) in input pixels.
    """
    if pred_xy_hm.shape != target_xy_hm.shape or pred_xy_hm.ndim != 2 or pred_xy_hm.shape[1] != 2:
        raise ValueError(f"Expected pred/target shape (B,2), got {pred_xy_hm.shape} vs {target_xy_hm.shape}")
    if valid.ndim != 1 or valid.shape[0] != pred_xy_hm.shape[0]:
        raise ValueError(f"Expected valid shape (B,), got {valid.shape}")
    stride = int(stride)
    if stride <= 0:
        raise ValueError("stride must be > 0")

    mask = valid > 0.5
    if mask.sum() == 0:
        return torch.tensor(float("nan"), device=pred_xy_hm.device, dtype=pred_xy_hm.dtype)
    mae_hm = (pred_xy_hm - target_xy_hm).abs().mean(dim=1)  # (B,) mean(|dx|,|dy|)
    return mae_hm[mask].mean() * float(stride)

def estimate_conv3d_max_batch(model: nn.Module, temporal_T: int, image_size: int, device: torch.device) -> int | None:
    """Estimate max per-process batch size before Conv3d hits int32 indexing limits.

    PyTorch can raise `RuntimeError: Input tensor is too large.` inside conv3d when
    intermediate activation tensors exceed ~2^31 elements (implementation-dependent).

    Returns:
        An integer batch size limit, or None if the model has no Conv3d modules.
    """
    conv3d_modules = [m for m in model.modules() if isinstance(m, nn.Conv3d)]
    if not conv3d_modules:
        return None

    max_in_numel = 0

    def hook_fn(module, inputs, output):
        nonlocal max_in_numel
        if not inputs:
            return
        x = inputs[0]
        if isinstance(x, torch.Tensor):
            max_in_numel = max(max_in_numel, int(x.numel()))

    hooks = [m.register_forward_hook(hook_fn) for m in conv3d_modules]
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros((1, 3, int(temporal_T), int(image_size), int(image_size)), device=device)
            _ = model(dummy)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)

    # Conservative int32 element indexing limit used by several CUDA kernels.
    int32_max = 2**31 - 1
    if max_in_numel <= 0:
        return None
    return max(1, int32_max // max_in_numel)


def combine_presence_probs(logits: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
    """Replica la logica di inferenza: media tra sigmoid(logit) e picco heatmap."""
    if heatmap.ndim == 4 and heatmap.shape[1] == 1:
        heatmap = heatmap.squeeze(1)
    probs = torch.sigmoid(logits).squeeze(1)
    peak = heatmap.amax(dim=[-1, -2])
    combined = 0.5 * probs + 0.5 * peak
    combined = torch.clamp(combined, 1e-6, 1 - 1e-6)  # per BCE numericamente stabile
    return combined

def log_batch_temporal_samples(batch, temporal_selector, tag="train", max_samples=3):
    T = temporal_selector.temporal_T
    if T <= 1:
        return
    paths = batch.get("image_path_abs") or batch.get("image_path")
    if paths is None:
        return
    if isinstance(paths, str):
        paths = [paths]
    count = min(max_samples, len(paths))
    print(f"[Debug] {tag}: showing {count} temporal samples (T={T}, stride={temporal_selector.temporal_stride})")
    for i in range(count):
        center_path = paths[i]
        window = temporal_selector.get_window(center_path)
        window_str = ", ".join(os.path.basename(p) for p in window)
        print(f"  sample {i}: {os.path.basename(center_path)} -> [{window_str}]")

def log_temporal_debug_samples(dataset, tag="train", max_samples=3):
    T = dataset.temporal_selector.temporal_T
    stride = dataset.temporal_selector.temporal_stride
    if T <= 1 or len(dataset) == 0:
        return
    step = max(1, len(dataset) // max_samples)
    indices = [min(i, len(dataset) - 1) for i in range(0, len(dataset), step)][:max_samples]
    print(f"[Debug] {tag} temporal samples (T={T}, stride={stride})")
    for idx in indices:
        row = dataset.df.iloc[idx]
        center_abs = row["image_path_abs"]
        window = dataset.temporal_selector.get_window(center_abs)
        window_str = ", ".join(os.path.basename(p) for p in window)
        print(f"  idx={idx}: {os.path.basename(row['image_path'])} -> [{window_str}]")


def evaluate_loader(model, loader, hm_loss, amp_enabled, loss_weights, device, rank: int = 0,
                    distributed: bool = False, world_size: int = 1,
                    presence_smoothing: float = 0.0,
                    presence_loss_fn=None,
                    log_combined_presence: bool = False,
                    input_key: str = "image",
                    presence_from_peak: bool = False,
                    heatmap_loss_type: str = "mse",
                    dsnt_tau: float = 1.0,
                    dsnt_coord_loss: str = "l1",
                    peak_pool: str = "max",
                    peak_tau: float = 1.0,
                    heatmap_stride: int = 4):
    #vL, vHm, vPr = [], [], []
    sum_L, sum_hm, sum_pr, sum_pr_comb = 0.0, 0.0, 0.0, 0.0
    sum_mae_px = 0.0
    n_mae = 0.0
    sum_pred_x = 0.0
    sum_pred_y = 0.0
    sum_pred_x2 = 0.0
    sum_pred_y2 = 0.0
    n_pred = 0.0
    sum_hm_focal_reg_num = 0.0
    sum_hm_focal_reg_den = 0.0
    bad_batches = 0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            total_batches += 1
            img = batch[input_key].to(device, non_blocking=True)
            hm_t = batch["heatmap"].to(device, non_blocking=True)
            pres = batch["presence"].to(device, non_blocking=True)
            pres_smooth = smooth_targets(pres, presence_smoothing)
            # Val in full precision to reduce risk of overflow/NaN
            with autocast(enabled=amp_enabled):
                hm_p, pres_logit = model(img)
                if hm_p.shape[-2:] != hm_t.shape[-2:]:
                    raise ValueError(
                        f"Heatmap shape mismatch: pred={tuple(hm_p.shape)} tgt={tuple(hm_t.shape)}. "
                        "Check heatmap_stride/config vs checkpoint/model."
                    )
                hm_p = torch.nan_to_num(hm_p, nan=0.0, posinf=50.0, neginf=-50.0).float()
                pres_logit = torch.nan_to_num(pres_logit, nan=0.0, posinf=50.0, neginf=-50.0).float()
                w_reg = float(loss_weights.get("w_heatmap_focal_reg", 0.0) or 0.0)
                L_reg = torch.tensor(0.0, device=device)
                if str(heatmap_loss_type).lower().strip() == "dsnt":
                    target_xy = batch["target_xy_hm"].to(device, non_blocking=True)
                    valid = batch["target_xy_valid"].to(device, non_blocking=True).squeeze(1)
                    pres_w = pres.squeeze(1).clamp(0.0, 1.0)
                    prob = spatial_softmax_2d(hm_p, tau=float(dsnt_tau))
                    pred_xy = dsnt_expectation(prob)
                    hm_per = coord_loss_per_sample(pred_xy, target_xy, dsnt_coord_loss)
                    pos_mult = float(loss_weights.get("heatmap_pos_multiplier", 1.0) or 1.0)
                    weight = pos_mult * valid * pres_w
                    L_hm = (hm_per * weight).sum() / torch.clamp(weight.sum(), min=1.0)
                    mae_px = center_mae_px(pred_xy, target_xy, valid, heatmap_stride)

                    mask = valid > 0.5
                    if mask.any():
                        pred_xy_px = pred_xy * float(heatmap_stride)
                        px = pred_xy_px[:, 0][mask]
                        py = pred_xy_px[:, 1][mask]
                        sum_pred_x += float(px.sum().item())
                        sum_pred_y += float(py.sum().item())
                        sum_pred_x2 += float((px ** 2).sum().item())
                        sum_pred_y2 += float((py ** 2).sum().item())
                        n_pred += float(mask.sum().item())

                    if w_reg > 0.0:
                        hm_focal_reg = HeatmapFocal(
                            alpha=float(loss_weights.get("heatmap_focal_alpha", 2.0) or 2.0),
                            beta=float(loss_weights.get("heatmap_focal_beta", 4.0) or 4.0),
                        )
                        focal_per = hm_focal_reg(hm_p, hm_t, reduction="none")  # (B,)
                        neg_mult_reg = float(loss_weights.get("heatmap_focal_reg_neg_multiplier", 0.0) or 0.0)
                        reg_w = valid * pres_w + neg_mult_reg * (1.0 - valid)
                        reg_w_sum = float(reg_w.sum().item())
                        if reg_w_sum > 0.0:
                            reg_num = float((focal_per * reg_w).sum().item())
                            sum_hm_focal_reg_num += reg_num
                            sum_hm_focal_reg_den += reg_w_sum
                            L_reg = (focal_per * reg_w).sum() / torch.clamp(reg_w.sum(), min=1.0)
                else:
                    hm_per = hm_loss(hm_p, hm_t, reduction="none")
                    pos_mask = (pres.squeeze(1) > 0.5).float()
                    neg_mask = 1.0 - pos_mask
                    neg_mult = float(loss_weights.get("heatmap_neg_multiplier", 1.0) or 1.0)
                    pos_mult = float(loss_weights.get("heatmap_pos_multiplier", 1.0) or 1.0)
                    weight = pos_mult * pos_mask + neg_mult * neg_mask
                    L_hm = (hm_per * weight).sum() / torch.clamp(weight.sum(), min=1.0)
                    mae_px = torch.tensor(float("nan"), device=device)
                if presence_from_peak:
                    peak = spatial_peak_pool(hm_p, pool=peak_pool, tau=float(peak_tau))
                    L_pr = nn.functional.binary_cross_entropy_with_logits(peak, pres_smooth.view_as(peak))
                    if log_combined_presence:
                        L_pr_comb = nn.functional.binary_cross_entropy_with_logits(peak, pres_smooth.view_as(peak))
                    else:
                        L_pr_comb = torch.tensor(0.0, device=device)
                else:
                    L_pr = presence_loss_fn(pres_logit, pres_smooth) if presence_loss_fn else bce_logits(pres_logit, pres_smooth)
                    if log_combined_presence:
                        comb_prob = combine_presence_probs(pres_logit, hm_p)
                        L_pr_comb = nn.functional.binary_cross_entropy(comb_prob, pres_smooth.view_as(comb_prob))
                    else:
                        L_pr_comb = torch.tensor(0.0, device=device)
                hm_w = float(loss_weights.get("w_heatmap", 1.0) or 1.0)
                pr_w_key = "w_peak_bce" if presence_from_peak else "w_presence"
                pr_w = float(loss_weights.get(pr_w_key, 1.0) or 1.0)
                L = hm_w * L_hm + pr_w * L_pr + w_reg * L_reg
            if not torch.isfinite(L):
                bad_batches += 1
                if bad_batches <= 3 and rank == 0:
                    print(f"[WARN] non-finite val loss (rank {rank}), skipping batch")
                continue
            sum_L += L.item(); sum_hm += L_hm.item(); sum_pr += L_pr.item(); sum_pr_comb += L_pr_comb.item()
            if torch.isfinite(mae_px):
                sum_mae_px += float(mae_px.item())
                n_mae += 1.0

    totals = torch.tensor(
        [
            sum_L, sum_hm, sum_pr, sum_pr_comb,
            sum_mae_px, n_mae,
            sum_pred_x, sum_pred_y, sum_pred_x2, sum_pred_y2, n_pred,
            sum_hm_focal_reg_num, sum_hm_focal_reg_den,
            float(total_batches), float(bad_batches),
        ],
        device=device
    )
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    (
        sum_L, sum_hm, sum_pr, sum_pr_comb,
        sum_mae_px, n_mae,
        sum_pred_x, sum_pred_y, sum_pred_x2, sum_pred_y2, n_pred,
        sum_hm_focal_reg_num, sum_hm_focal_reg_den,
        total_batches, bad_batches,
    ) = totals.tolist()
    if total_batches - bad_batches > 0:
        mean_L = sum_L / (total_batches - bad_batches)
        mean_hm = sum_hm / (total_batches - bad_batches)
        mean_pr = sum_pr / (total_batches - bad_batches)
        mean_pr_comb = sum_pr_comb / (total_batches - bad_batches)
        mean_mae_px = (sum_mae_px / n_mae) if n_mae > 0 else float("nan")
        pred_x_mean = (sum_pred_x / n_pred) if n_pred > 0 else float("nan")
        pred_y_mean = (sum_pred_y / n_pred) if n_pred > 0 else float("nan")
        pred_x_var = (sum_pred_x2 / n_pred - pred_x_mean ** 2) if n_pred > 0 else float("nan")
        pred_y_var = (sum_pred_y2 / n_pred - pred_y_mean ** 2) if n_pred > 0 else float("nan")
        hm_focal_reg = (sum_hm_focal_reg_num / sum_hm_focal_reg_den) if sum_hm_focal_reg_den > 0 else float("nan")
    else:
        mean_L = mean_hm = mean_pr = mean_pr_comb = mean_mae_px = float("nan")
        pred_x_mean = pred_y_mean = pred_x_var = pred_y_var = float("nan")
        hm_focal_reg = float("nan")
    return {
        "loss": float(mean_L),
        "hm": float(mean_hm),
        "presence": float(mean_pr),
        "presence_combined": float(mean_pr_comb) if log_combined_presence else None,
        "center_mae_px": float(mean_mae_px),
        "pred_x_mean_px": float(pred_x_mean),
        "pred_x_var_px": float(pred_x_var),
        "pred_y_mean_px": float(pred_y_mean),
        "pred_y_var_px": float(pred_y_var),
        "heatmap_focal_reg": float(hm_focal_reg),
        "bad_batches": int(bad_batches),
        "total_batches": int(total_batches)
    }

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    # Optional CLI overrides keep backwards-compat with config-driven flow.
    if args.train_csv:
        cfg["data"]["manifest_train"] = args.train_csv
    if args.val_csv:
        cfg["data"]["manifest_val"] = args.val_csv
    if args.test_csv:
        cfg["data"]["manifest_test"] = args.test_csv
    if args.image_size:
        cfg["train"]["image_size"] = args.image_size
    if args.heatmap_stride:
        cfg["train"]["heatmap_stride"] = args.heatmap_stride
    if args.heatmap_sigma_px:
        cfg["loss"]["heatmap_sigma_px"] = args.heatmap_sigma_px
    if args.heatmap_loss:
        cfg["loss"]["heatmap_loss"] = args.heatmap_loss
    if args.dsnt_tau is not None:
        cfg["loss"]["dsnt_tau"] = float(args.dsnt_tau)
    if args.dsnt_coord_loss:
        cfg["loss"]["dsnt_coord_loss"] = str(args.dsnt_coord_loss)
    if args.peak_pool:
        cfg["loss"]["peak_pool"] = str(args.peak_pool)
    if args.peak_tau is not None:
        cfg["loss"]["peak_tau"] = float(args.peak_tau)
    if args.w_heatmap_focal_reg is not None:
        cfg["loss"]["w_heatmap_focal_reg"] = float(args.w_heatmap_focal_reg)
    if args.heatmap_focal_reg_neg_multiplier is not None:
        cfg["loss"]["heatmap_focal_reg_neg_multiplier"] = float(args.heatmap_focal_reg_neg_multiplier)
    if args.heatmap_neg_multiplier is not None:
        cfg["loss"]["heatmap_neg_multiplier"] = args.heatmap_neg_multiplier
    if args.heatmap_pos_multiplier is not None:
        cfg["loss"]["heatmap_pos_multiplier"] = args.heatmap_pos_multiplier
    if args.presence_from_peak is not None:
        cfg["train"]["presence_from_peak"] = bool(args.presence_from_peak)
    if args.backbone:
        cfg["train"]["backbone"] = args.backbone
    if args.temporal_T:
        cfg["train"]["temporal_T"] = args.temporal_T
    if args.temporal_stride:
        cfg["train"]["temporal_stride"] = args.temporal_stride
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.bs:
        cfg["train"]["batch_size"] = args.bs
    if args.lr:
        cfg["train"]["lr"] = args.lr
    if args.log_dir:
        cfg["train"]["save_dir"] = args.log_dir
    if args.best_ckpt_start_epoch is not None:
        cfg["train"]["best_ckpt_start_epoch"] = args.best_ckpt_start_epoch
    if args.grad_accum_steps is not None:
        cfg["train"]["grad_accum_steps"] = args.grad_accum_steps
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.dataloader_timeout_s is not None:
        cfg["train"]["dataloader_timeout_s"] = args.dataloader_timeout_s
    if args.persistent_workers is not None:
        cfg["train"]["persistent_workers"] = bool(args.persistent_workers)

    # temporal parameters
    temporal_T = max(1, int(cfg["train"].get("temporal_T", 1)))
    temporal_stride = max(1, int(cfg["train"].get("temporal_stride", 1)))
    cfg["train"]["temporal_T"] = temporal_T
    cfg["train"]["temporal_stride"] = temporal_stride
    print(f"Temporal window: T={temporal_T}, stride={temporal_stride}")


    # Distributed setup
    rank, local_rank, world_size, _, env_num_workers = get_resources()
    distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if distributed and device.type == "cuda":
        torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            init_method="env://",
            rank=rank,
            world_size=world_size
        )
        dist.barrier()

    set_seed(cfg["train"]["seed"] + rank)

    configured_workers = cfg["train"].get("num_workers", env_num_workers)
    num_workers = configured_workers if configured_workers is not None else env_num_workers
    if env_num_workers and num_workers is not None:
        num_workers = min(num_workers, env_num_workers)
    num_workers = int(num_workers or 0)
    cfg["train"]["num_workers"] = num_workers
    dl_timeout = int(cfg["train"].get("dataloader_timeout_s", 0) or 0)
    # PyTorch requires timeout=0 when num_workers=0 (single-process data loading).
    loader_timeout = dl_timeout if num_workers > 0 else 0
    persistent_workers_cfg = bool(cfg["train"].get("persistent_workers", True))
    persistent_workers = num_workers > 0 and persistent_workers_cfg
    pin_memory = device.type == "cuda"

    

    # Datasets
    ds_tr = MedFullBasinDataset(
        cfg["data"]["manifest_train"],
        image_size=cfg["train"]["image_size"],
        heatmap_stride=cfg["train"]["heatmap_stride"],
        heatmap_sigma_px=cfg["loss"]["heatmap_sigma_px"],
        use_aug=cfg["train"]["use_aug"],
        use_pre_letterboxed=cfg["data"]["use_pre_letterboxed"],
        letterbox_meta_csv=cfg["data"]["letterbox_meta_csv"],
        letterbox_size_assert=cfg["data"]["letterbox_size_assert"],
        temporal_T=temporal_T,
        temporal_stride=temporal_stride,
        manifest_stride=int(cfg["data"].get("manifest_stride", 1) or 1),
    )
    #log_temporal_debug_samples(ds_tr, tag="train")
    ds_va = MedFullBasinDataset(
        cfg["data"]["manifest_val"],
        image_size=cfg["train"]["image_size"],
        heatmap_stride=cfg["train"]["heatmap_stride"],
        heatmap_sigma_px=cfg["loss"]["heatmap_sigma_px"],
        use_aug=False,
        use_pre_letterboxed=cfg["data"]["use_pre_letterboxed"],
        letterbox_meta_csv=cfg["data"]["letterbox_meta_csv"],
        letterbox_size_assert=cfg["data"]["letterbox_size_assert"],
        temporal_T=temporal_T,
        temporal_stride=temporal_stride,
        manifest_stride=int(cfg["data"].get("manifest_stride", 1) or 1),
    )

    test_loader = None
    manifest_test = cfg["data"].get("manifest_test")
    if manifest_test:
        if os.path.exists(manifest_test):
            ds_te = MedFullBasinDataset(
                manifest_test,
                image_size=cfg["train"]["image_size"],
                heatmap_stride=cfg["train"]["heatmap_stride"],
                heatmap_sigma_px=cfg["loss"]["heatmap_sigma_px"],
                use_aug=False,
                use_pre_letterboxed=cfg["data"]["use_pre_letterboxed"],
                letterbox_meta_csv=cfg["data"]["letterbox_meta_csv"],
                letterbox_size_assert=cfg["data"]["letterbox_size_assert"],
                temporal_T=temporal_T,
                temporal_stride=temporal_stride,
                manifest_stride=int(cfg["data"].get("manifest_stride", 1) or 1),
            )

            test_sampler = DistributedSampler(
                ds_te,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False
            ) if distributed else None
            test_loader = DataLoader(
                ds_te,
                batch_size=cfg["train"]["batch_size"],
                shuffle=test_sampler is None,
                sampler=test_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                worker_init_fn=seed_worker if num_workers > 0 else None
            )
        elif is_main_process(rank):
            print(f"[WARN] Test manifest {manifest_test} non trovato: salto l'eval di test.")

    train_sampler = DistributedSampler(
        ds_tr,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    ) if distributed else None

    tr_loader = DataLoader(
        ds_tr,
        batch_size=cfg["train"]["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        timeout=loader_timeout,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        drop_last=True
    )
    va_sampler = DistributedSampler(
        ds_va,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    ) if distributed else None

    va_loader = DataLoader(
        ds_va,
        batch_size=cfg["train"]["batch_size"],
        shuffle=va_sampler is None,
        sampler=va_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        timeout=loader_timeout,
        worker_init_fn=seed_worker if num_workers > 0 else None
    )

    presence_from_peak = bool(cfg["train"].get("presence_from_peak", False))

    # Model
    backbone_name = cfg["train"]["backbone"]
    pretrained_backbone = bool(cfg["train"].get("backbone_pretrained", True))
    if backbone_name.startswith("x3d"):
        model = X3DBackbone(
            backbone=backbone_name,
            out_heatmap_ch=1,
            presence_dropout=cfg["train"].get("presence_dropout", 0.0),
            pretrained=pretrained_backbone,
            heatmap_stride=int(cfg["train"]["heatmap_stride"]),
        )
    else:
        model = SimpleBaseline(
            backbone=backbone_name,
            out_heatmap_ch=1,
            temporal_T=temporal_T,
            presence_dropout=cfg["train"].get("presence_dropout", 0.0),
            pretrained=pretrained_backbone,
            heatmap_stride=int(cfg["train"]["heatmap_stride"]),
        )
    model = model.to(device)

    # Preflight: avoid opaque conv3d "Input tensor is too large" crashes (common with large per-GPU batches).
    if backbone_name.startswith("x3d"):
        max_bs = None
        if is_main_process(rank):
            try:
                max_bs = estimate_conv3d_max_batch(model, temporal_T=temporal_T, image_size=cfg["train"]["image_size"], device=device)
            except Exception as exc:
                print(f"[WARN] could not estimate Conv3d safe batch size; continuing (reason: {exc})")
                max_bs = None
        if distributed:
            max_bs_t = torch.tensor([-1 if max_bs is None else int(max_bs)], device=device, dtype=torch.int64)
            dist.broadcast(max_bs_t, src=0)
            max_bs = None if int(max_bs_t.item()) < 0 else int(max_bs_t.item())

        if max_bs is not None and int(cfg["train"]["batch_size"]) > max_bs:
            if is_main_process(rank):
                print(
                    f"[ERROR] x3d backbone conv3d activations exceed int32 indexing limits with batch_size={cfg['train']['batch_size']}.\n"
                    f"        Estimated max batch_size per process: {max_bs} (for T={temporal_T}, image_size={cfg['train']['image_size']}).\n"
                    f"        Fix: reduce `train.batch_size` (or `temporal_T` / `image_size`), or add grad accumulation."
                )
            raise ValueError("batch_size too large for Conv3d kernels (see log above)")

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=presence_from_peak  # presence head non usata se si usa solo il picco
        )
    model_to_save = model.module if distributed else model
    input_key = "video" if getattr(model_to_save, "input_is_video", False) else "image"


    # Optim
    base_lr = cfg["train"]["lr"]
    scaled_lr = base_lr * world_size if (distributed and cfg["train"].get("scale_lr_by_world_size", True)) else base_lr
    opt = torch.optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=cfg["train"]["weight_decay"])
    scaler = GradScaler(enabled=cfg["train"]["amp"] and device.type == "cuda")
    heatmap_loss_type = cfg["loss"].get("heatmap_loss", "mse")
    dsnt_tau = float(cfg["loss"].get("dsnt_tau", 1.0) or 1.0)
    dsnt_coord_loss = str(cfg["loss"].get("dsnt_coord_loss", "l1") or "l1")
    peak_pool = str(cfg["loss"].get("peak_pool", "max") or "max")
    peak_tau = float(cfg["loss"].get("peak_tau", 1.0) or 1.0)
    if str(heatmap_loss_type).lower().strip() == "dsnt":
        hm_loss = None
    elif heatmap_loss_type == "focal":
        hm_loss = HeatmapFocal(
            alpha=float(cfg["loss"].get("heatmap_focal_alpha", 2.0) or 2.0),
            beta=float(cfg["loss"].get("heatmap_focal_beta", 4.0) or 4.0),
        )
    else:
        hm_loss = HeatmapMSE()
    loss_weights = cfg["loss"]
    hm_focal_struct = HeatmapFocal(
        alpha=float(cfg["loss"].get("heatmap_focal_alpha", 2.0) or 2.0),
        beta=float(cfg["loss"].get("heatmap_focal_beta", 4.0) or 4.0),
    )
    w_hm_focal_reg = float(cfg["loss"].get("w_heatmap_focal_reg", 0.0) or 0.0)
    hm_focal_reg_neg_mult = float(cfg["loss"].get("heatmap_focal_reg_neg_multiplier", 0.0) or 0.0)
    presence_smoothing = float(cfg["loss"].get("presence_label_smoothing", 0.0) or 0.0)
    presence_loss_type = cfg["loss"].get("presence_loss", "bce")
    focal_gamma = float(cfg["loss"].get("presence_focal_gamma", 2.0) or 2.0)
    focal_alpha = cfg["loss"].get("presence_focal_alpha", None)
    if focal_alpha is not None:
        focal_alpha = float(focal_alpha)
    if presence_loss_type == "focal":
        def presence_loss_fn(logits, target):
            return focal_loss_with_logits(logits, target, gamma=focal_gamma, alpha=focal_alpha)
    else:
        def presence_loss_fn(logits, target):
            return bce_logits(logits, target)

    if is_main_process(rank):
        save_dir = cfg["train"]["save_dir"]; os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = cfg["train"]["save_dir"]
    if distributed:
        dist.barrier()

    log_path = os.path.join(save_dir, "training_log.csv")
    log_fields = [
        "epoch",
        "train_loss", "train_heatmap_loss", "train_presence_loss",
        "train_center_mae_px",
        "train_heatmap_focal_reg",
        "val_loss", "val_heatmap_loss", "val_presence_loss",
        "val_center_mae_px",
        "val_heatmap_focal_reg",
        "test_loss", "test_heatmap_loss", "test_presence_loss", "test_presence_combined_loss"
    ]
    log_fields.append("test_center_mae_px")
    log_fields.append("test_heatmap_focal_reg")
    log_fields += [
        "val_pred_x_mean_px", "val_pred_x_var_px", "val_pred_y_mean_px", "val_pred_y_var_px",
        "test_pred_x_mean_px", "test_pred_x_var_px", "test_pred_y_mean_px", "test_pred_y_var_px",
    ]
    if is_main_process(rank):
        print(f"[INFO] rank {rank}/{world_size} device={device}")
        print(f"[INFO] batch_size per GPU={cfg['train']['batch_size']} (global={cfg['train']['batch_size'] * world_size})")
        if scaled_lr != base_lr:
            print(f"[INFO] lr scaled from {base_lr} to {scaled_lr} for world_size={world_size}")
        print(f"Logging metrics to {log_path}")

    best_val = 1e9; best_path = None
    best_start_epoch = cfg["train"].get("best_ckpt_start_epoch", 1)
    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1) or 1)
    if grad_accum_steps < 1:
        grad_accum_steps = 1

    log_file = None
    writer = None
    if is_main_process(rank):
        log_file = open(log_path, "w", newline="")
        writer = csv.DictWriter(log_file, fieldnames=log_fields)
        writer.writeheader()

    try:
        for epoch in range(1, cfg["train"]["epochs"]+1):
            epoch_start = time.time()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            losses = []; hm_losses = []; pres_losses = []; peak_preds = []; mae_center_px = []; hm_reg_losses = []
            opt.zero_grad(set_to_none=True)
            last_micro_step = -1
            for batch_idx, batch in enumerate(tr_loader):
                #log_batch_temporal_samples(batch, ds_tr.temporal_selector, tag=f"train batch ", max_samples=3)   #{batch_idx}
                img = batch[input_key].to(device, non_blocking=True)
                hm_t = batch["heatmap"].to(device, non_blocking=True)
                pres = batch["presence"].to(device, non_blocking=True)
                pres_smooth = smooth_targets(pres, presence_smoothing)
                pres_raw = pres.squeeze(1)

                micro_step = batch_idx % grad_accum_steps
                last_micro_step = micro_step
                sync_ctx = nullcontext()
                if distributed and micro_step != (grad_accum_steps - 1):
                    sync_ctx = model.no_sync()  # type: ignore[attr-defined]

                with sync_ctx:
                    with autocast(enabled=cfg["train"]["amp"]):
                        hm_p, pres_logit = model(img)
                        if hm_p.shape[-2:] != hm_t.shape[-2:]:
                            raise ValueError(
                                f"Heatmap shape mismatch: pred={tuple(hm_p.shape)} tgt={tuple(hm_t.shape)}. "
                                "Check heatmap_stride/config vs checkpoint/model."
                            )
                        # Stabilize numeric range under AMP (especially with logsumexp pooling and small tau).
                        hm_p = torch.nan_to_num(hm_p, nan=0.0, posinf=50.0, neginf=-50.0).float()
                        pres_logit = torch.nan_to_num(pres_logit, nan=0.0, posinf=50.0, neginf=-50.0).float()
                        if str(heatmap_loss_type).lower().strip() == "dsnt":
                            target_xy = batch["target_xy_hm"].to(device, non_blocking=True)
                            valid = batch["target_xy_valid"].to(device, non_blocking=True).squeeze(1)
                            pres_w = pres_raw.clamp(0.0, 1.0)
                            prob = spatial_softmax_2d(hm_p, tau=float(dsnt_tau))
                            pred_xy = dsnt_expectation(prob)
                            hm_per = coord_loss_per_sample(pred_xy, target_xy, dsnt_coord_loss)
                            pos_mult = float(cfg["loss"].get("heatmap_pos_multiplier", 1.0) or 1.0)
                            weight = pos_mult * valid * pres_w
                            L_hm = (hm_per * weight).sum() / torch.clamp(weight.sum(), min=1.0)
                            mae_px = center_mae_px(pred_xy, target_xy, valid, cfg["train"]["heatmap_stride"])
                            L_reg = torch.tensor(0.0, device=device)
                            if w_hm_focal_reg > 0.0:
                                focal_per = hm_focal_struct(hm_p, hm_t, reduction="none")
                                reg_w = valid * pres_w + hm_focal_reg_neg_mult * (1.0 - valid)
                                if float(reg_w.sum().item()) > 0.0:
                                    L_reg = (focal_per * reg_w).sum() / torch.clamp(reg_w.sum(), min=1.0)
                        else:
                            hm_per = hm_loss(hm_p, hm_t, reduction="none")
                            pos_mask = (pres_raw > 0.5).float()
                            neg_mask = 1.0 - pos_mask
                            neg_mult = float(cfg["loss"].get("heatmap_neg_multiplier", 1.0) or 1.0)
                            pos_mult = float(cfg["loss"].get("heatmap_pos_multiplier", 1.0) or 1.0)
                            weight = pos_mult * pos_mask + neg_mult * neg_mask
                            L_hm = (hm_per * weight).sum() / torch.clamp(weight.sum(), min=1.0)
                            mae_px = torch.tensor(float("nan"), device=device)
                            L_reg = torch.tensor(0.0, device=device)
                        if presence_from_peak:
                            peak = spatial_peak_pool(hm_p, pool=peak_pool, tau=float(peak_tau))
                            L_pr = nn.functional.binary_cross_entropy_with_logits(peak, pres_smooth.view_as(peak))
                        else:
                            L_pr = presence_loss_fn(pres_logit, pres_smooth)
                        hm_w = float(loss_weights.get("w_heatmap", 1.0) or 1.0)
                        pr_w_key = "w_peak_bce" if presence_from_peak else "w_presence"
                        pr_w = float(loss_weights.get(pr_w_key, 1.0) or 1.0)
                        L = hm_w * L_hm + pr_w * L_pr + float(w_hm_focal_reg) * L_reg
                if not torch.isfinite(L):
                    if is_main_process(rank):
                        print(f"[ERROR] non-finite train loss at epoch {epoch}; aborting to avoid hang.")
                    raise RuntimeError("Non-finite training loss")
                scaler.scale(L / grad_accum_steps).backward()
                if micro_step == (grad_accum_steps - 1):
                    scaler.step(opt); scaler.update()
                    opt.zero_grad(set_to_none=True)

                losses.append(L.item()); hm_losses.append(L_hm.item()); pres_losses.append(L_pr.item())
                peak_preds.append(spatial_peak_pool(hm_p.detach(), pool=peak_pool, tau=float(peak_tau)).mean().item())
                if torch.isfinite(mae_px):
                    mae_center_px.append(float(mae_px.item()))
                if torch.isfinite(L_reg) and float(w_hm_focal_reg) > 0.0:
                    hm_reg_losses.append(float(L_reg.item()))

            if last_micro_step != -1 and last_micro_step != (grad_accum_steps - 1):
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)

            tr_loss = float(np.mean(losses)) if losses else 0.0
            tr_hm = float(np.mean(hm_losses)) if hm_losses else 0.0
            tr_pr = float(np.mean(pres_losses)) if pres_losses else 0.0
            tr_peak = float(np.mean(peak_preds)) if peak_preds else 0.0
            tr_mae_px = float(np.mean(mae_center_px)) if mae_center_px else float("nan")
            tr_hm_reg = float(np.mean(hm_reg_losses)) if hm_reg_losses else float("nan")

            metrics_tensor = torch.tensor([tr_loss, tr_hm, tr_pr, tr_peak, tr_mae_px, tr_hm_reg], device=device)
            metrics_tensor = reduce_mean(metrics_tensor)
            tr_loss, tr_hm, tr_pr, tr_peak, tr_mae_px, tr_hm_reg = [float(x) for x in metrics_tensor.tolist()]

            if is_main_process(rank):
                msg = f"[Epoch {epoch}] train: L={tr_loss:.4f} (hm={tr_hm:.4f}, pr={tr_pr:.4f}, peak={tr_peak:.4f})"
                if np.isfinite(tr_mae_px):
                    msg += f", mae_px={tr_mae_px:.2f}"
                if np.isfinite(tr_hm_reg) and w_hm_focal_reg > 0.0:
                    msg += f", hm_reg={tr_hm_reg:.4f}"
                print(msg)

            val_metrics = None; test_metrics = None
            if epoch % cfg["train"]["val_every"] == 0:
                if distributed:
                    dist.barrier()

                eval_model = model.module if distributed else model
                eval_model.eval()
                val_metrics = evaluate_loader(
                    eval_model, va_loader, hm_loss, False, loss_weights, device,
                    rank=rank, distributed=distributed, world_size=world_size,
                    presence_smoothing=presence_smoothing,
                    presence_loss_fn=presence_loss_fn,
                    log_combined_presence=False,
                    input_key=input_key,
                    presence_from_peak=presence_from_peak,
                    heatmap_loss_type=heatmap_loss_type,
                    dsnt_tau=dsnt_tau,
                    dsnt_coord_loss=dsnt_coord_loss,
                    peak_pool=peak_pool,
                    peak_tau=peak_tau,
                    heatmap_stride=cfg["train"]["heatmap_stride"],
                )
                if test_loader is not None:
                    test_metrics = evaluate_loader(
                        eval_model, test_loader, hm_loss, False, loss_weights, device,
                        rank=rank, distributed=distributed, world_size=world_size,
                        presence_smoothing=presence_smoothing,
                        presence_loss_fn=presence_loss_fn,
                        log_combined_presence=True,
                        input_key=input_key,
                        presence_from_peak=presence_from_peak,
                        heatmap_loss_type=heatmap_loss_type,
                        dsnt_tau=dsnt_tau,
                        dsnt_coord_loss=dsnt_coord_loss,
                        peak_pool=peak_pool,
                        peak_tau=peak_tau,
                        heatmap_stride=cfg["train"]["heatmap_stride"],
                    )
                eval_model.train()

                if distributed:
                    dist.barrier()

                if is_main_process(rank) and val_metrics is not None:
                    val_score = val_metrics["loss"]
                    if val_metrics.get("bad_batches", 0) > 0:
                        total = val_metrics.get("total_batches", 0)
                        msg = f"[WARN] val skipped {val_metrics['bad_batches']} non-finite batches"
                        if total:
                            msg += f" out of {total}"
                        print(msg)
                        if total and val_metrics["bad_batches"] == total:
                            raise RuntimeError("Validation produced only non-finite batches; aborting.")
                    msg = f"          val:  L={val_score:.4f} (hm={val_metrics['hm']:.4f}, pr={val_metrics['presence']:.4f})"
                    if np.isfinite(val_metrics.get('center_mae_px', float('nan'))):
                        msg += f", mae_px={val_metrics['center_mae_px']:.2f}"
                    if np.isfinite(val_metrics.get("heatmap_focal_reg", float("nan"))) and w_hm_focal_reg > 0.0:
                        msg += f", hm_reg={val_metrics['heatmap_focal_reg']:.4f}"
                    print(msg)
                    if np.isfinite(val_metrics.get("pred_x_mean_px", float("nan"))):
                        print(
                            f"          val:  pred_xy mean=({val_metrics['pred_x_mean_px']:.2f}, {val_metrics['pred_y_mean_px']:.2f}) "
                            f"var=({val_metrics['pred_x_var_px']:.2f}, {val_metrics['pred_y_var_px']:.2f}) px"
                        )
                    if test_metrics is not None and np.isfinite(test_metrics.get("pred_x_mean_px", float("nan"))):
                        print(
                            f"          test: pred_xy mean=({test_metrics['pred_x_mean_px']:.2f}, {test_metrics['pred_y_mean_px']:.2f}) "
                            f"var=({test_metrics['pred_x_var_px']:.2f}, {test_metrics['pred_y_var_px']:.2f}) px"
                        )

                    if epoch < best_start_epoch:
                        print(f"[Epoch {epoch}] checkpoint skip: epoch < best_ckpt_start_epoch ({best_start_epoch})")
                    elif val_score < best_val:
                        best_val = val_score
                        best_target = os.path.join(save_dir, "best.ckpt")
                        t0 = time.time()
                        print(f"[Epoch {epoch}] saving BEST checkpoint to {best_target} ...")
                        torch.save({"model": model_to_save.state_dict(), "cfg": cfg}, best_target)
                        print(f"[Epoch {epoch}] best checkpoint saved in {time.time()-t0:.2f}s")
                        best_path = best_target
                    else:
                        print(f"[Epoch {epoch}] no val improvement ({val_score:.4f} >= {best_val:.4f}); checkpoint not saved")

            if writer is not None:
                row = {
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "train_heatmap_loss": tr_hm,
                    "train_presence_loss": tr_pr,
                    "train_center_mae_px": "" if not np.isfinite(tr_mae_px) else tr_mae_px,
                    "train_heatmap_focal_reg": "" if not np.isfinite(tr_hm_reg) else tr_hm_reg,
                    "val_loss": val_metrics["loss"] if val_metrics else "",
                    "val_heatmap_loss": val_metrics["hm"] if val_metrics else "",
                    "val_presence_loss": val_metrics["presence"] if val_metrics else "",
                    "val_center_mae_px": val_metrics.get("center_mae_px", "") if val_metrics else "",
                    "val_heatmap_focal_reg": val_metrics.get("heatmap_focal_reg", "") if val_metrics else "",
                    "test_loss": test_metrics["loss"] if test_metrics else "",
                    "test_heatmap_loss": test_metrics["hm"] if test_metrics else "",
                    "test_presence_loss": test_metrics["presence"] if test_metrics else "",
                    "test_presence_combined_loss": test_metrics["presence_combined"] if test_metrics else "",
                    "test_center_mae_px": test_metrics.get("center_mae_px", "") if test_metrics else "",
                    "test_heatmap_focal_reg": test_metrics.get("heatmap_focal_reg", "") if test_metrics else "",
                    "val_pred_x_mean_px": val_metrics.get("pred_x_mean_px", "") if val_metrics else "",
                    "val_pred_x_var_px": val_metrics.get("pred_x_var_px", "") if val_metrics else "",
                    "val_pred_y_mean_px": val_metrics.get("pred_y_mean_px", "") if val_metrics else "",
                    "val_pred_y_var_px": val_metrics.get("pred_y_var_px", "") if val_metrics else "",
                    "test_pred_x_mean_px": test_metrics.get("pred_x_mean_px", "") if test_metrics else "",
                    "test_pred_x_var_px": test_metrics.get("pred_x_var_px", "") if test_metrics else "",
                    "test_pred_y_mean_px": test_metrics.get("pred_y_mean_px", "") if test_metrics else "",
                    "test_pred_y_var_px": test_metrics.get("pred_y_var_px", "") if test_metrics else "",
                }
                writer.writerow(row); log_file.flush()

            if distributed:
                dist.barrier()
            elapsed = time.time() - epoch_start
            if is_main_process(rank):
                print(f"[Epoch {epoch}] elapsed: {elapsed:.1f}s")
    finally:
        if log_file is not None:
            log_file.close()

    if is_main_process(rank):
        print("Done. Best:", best_path)
    cleanup_distributed()

if __name__ == "__main__":
    main()
