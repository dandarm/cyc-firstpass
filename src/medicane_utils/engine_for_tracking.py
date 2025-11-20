"""Training utilities for cyclone center tracking."""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from medicane_utils.geo_const import get_lon_lat_grid_2_pixel

import utils


IMAGE_WIDTH = 1290
IMAGE_HEIGHT = 420
_TILE_OFFSETS_RE = re.compile(r"_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$")
EARTH_RADIUS_KM = 6371.0088


@lru_cache(maxsize=1)
def _get_lon_lat_grid() -> tuple[np.ndarray, np.ndarray]:
    lon_grid, lat_grid, _, _ = get_lon_lat_grid_2_pixel(IMAGE_WIDTH, IMAGE_HEIGHT)
    return lon_grid, lat_grid


def _pixels_to_latlon(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon_grid, lat_grid = _get_lon_lat_grid()
    x_idx = np.clip(np.rint(x).astype(int), 0, IMAGE_WIDTH - 1)
    y_idx = np.clip(np.rint(y).astype(int), 0, IMAGE_HEIGHT - 1)
    row_idx = IMAGE_HEIGHT - 1 - y_idx
    lat = lat_grid[row_idx, x_idx]
    lon = lon_grid[row_idx, x_idx]
    return lat, lon


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def _parse_tile_offsets(path: str) -> tuple[float, float]:
    base = os.path.basename(str(path).rstrip("/\\"))
    match = _TILE_OFFSETS_RE.search(base)
    if match is None:
        raise ValueError(f"Unable to parse tile offsets from path: {path}")
    return float(match.group(1)), float(match.group(2))


def batch_geo_distance_km(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    paths: Sequence[str],
) -> Optional[float]:
    if len(paths) == 0:
        return None
    if predictions.numel() == 0 or targets.numel() == 0:
        return None

    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    if not (np.isfinite(pred_np).all() and np.isfinite(target_np).all()):
        return None

    try:
        offsets = np.stack([_parse_tile_offsets(p) for p in paths], dtype=np.float32)
    except ValueError as exc:
        print(f"[WARN][tracking] {exc}")
        return None

    global_pred = pred_np + offsets
    global_target = target_np + offsets

    lat_pred, lon_pred = _pixels_to_latlon(global_pred[:, 0], global_pred[:, 1])
    lat_true, lon_true = _pixels_to_latlon(global_target[:, 0], global_target[:, 1])

    distances = _haversine_km(lat_pred, lon_pred, lat_true, lon_true)
    finite_mask = np.isfinite(distances)
    if not finite_mask.any():
        return None
    return float(distances[finite_mask].mean())


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    log_writer: Optional[utils.TensorboardLogger] = None,
    *,
    start_steps: Optional[int] = None,
    lr_schedule_values: Optional[Sequence[float]] = None,
    wd_schedule_values: Optional[Sequence[float]] = None,
    num_training_steps_per_epoch: Optional[int] = None,
) -> dict:
    """Train for a single epoch."""
    # ensure different shuffles across workers when using DistributedSampler
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = f"Epoch: [{epoch}]"
    if num_training_steps_per_epoch is not None and start_steps is None:
        start_steps = epoch * num_training_steps_per_epoch
    for batch_idx, (samples, target, paths) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        if num_training_steps_per_epoch is not None and batch_idx >= num_training_steps_per_epoch:
            continue

        global_step = None
        if start_steps is not None:
            global_step = start_steps + batch_idx
        elif num_training_steps_per_epoch is not None:
            global_step = epoch * num_training_steps_per_epoch + batch_idx

        lr_index = None
        if lr_schedule_values is not None and global_step is not None and len(lr_schedule_values) > 0:
            lr_index = min(global_step, len(lr_schedule_values) - 1)
        wd_index = None
        if wd_schedule_values is not None and global_step is not None and len(wd_schedule_values) > 0:
            wd_index = min(global_step, len(wd_schedule_values) - 1)

        if lr_index is not None or wd_index is not None:
            for param_group in optimizer.param_groups:
                if lr_index is not None:
                    lr_scale = param_group.get("lr_scale", 1.0)
                    param_group["lr"] = lr_schedule_values[lr_index] * lr_scale
                if wd_index is not None and param_group.get("weight_decay", 0) > 0:
                    param_group["weight_decay"] = wd_schedule_values[wd_index]

        samples = samples.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Guard: check inputs/targets are finite
        if not torch.isfinite(samples).all():
            print(f"[WARN] Non-finite samples at batch {batch_idx}: nan={torch.isnan(samples).sum().item()}, inf={torch.isinf(samples).sum().item()}")
        if not torch.isfinite(target).all():
            print(f"[WARN] Non-finite target at batch {batch_idx}: values={target}")

        output = model(samples)
        # Shape sanity check
        if output.ndim != target.ndim or output.shape[-1] != target.shape[-1]:
            try:
                print(f"[WARN] Output/target shape mismatch at batch {batch_idx}: output={tuple(output.shape)}, target={tuple(target.shape)}")
            except Exception:
                pass

        # Guard: check model outputs are finite
        if not torch.isfinite(output).all():
            o_nan = torch.isnan(output).sum().item()
            o_inf = torch.isinf(output).sum().item()
            o_min = float(torch.nanmin(output).detach().cpu()) if o_nan == 0 else float('nan')
            o_max = float(torch.nanmax(output).detach().cpu()) if o_nan == 0 else float('nan')
            print(f"[WARN] Non-finite output at batch {batch_idx}: nan={o_nan}, inf={o_inf}, min={o_min}, max={o_max}")

        loss = criterion(output, target)
        loss_value = loss.item()

        # Guard: skip update on non-finite loss to avoid poisoning training
        if not torch.isfinite(loss):
            t_min = float(torch.nanmin(target).detach().cpu()) if torch.isfinite(target).any() else float('nan')
            t_max = float(torch.nanmax(target).detach().cpu()) if torch.isfinite(target).any() else float('nan')
            print(f"[ERROR] Non-finite loss at batch {batch_idx}: loss={loss_value if loss.numel()==1 else loss}\n"
                  f"        target[min,max]=({t_min:.4f},{t_max:.4f}), paths[0]={paths[0] if isinstance(paths, (list, tuple)) and len(paths)>0 else paths}")
            optimizer.zero_grad(set_to_none=True)
            continue

        geo_err = batch_geo_distance_km(output, target, paths)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()



        lr_values = [group["lr"] for group in optimizer.param_groups if "lr" in group]
        max_lr = max(lr_values) if lr_values else None
        metric_logger.update(loss=loss_value, geo_km=geo_err, lr=max_lr)
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            if lr_values:
                log_writer.update(lr=max_lr, head="opt")
            if geo_err is not None:
                log_writer.update(geo_km=geo_err, head="metrics")
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
) -> dict:
    """Evaluate the model."""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    for batch_idx, (samples, target, paths) in enumerate(metric_logger.log_every(data_loader, 20, header)):
        samples = samples.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(samples)
        loss = criterion(output, target)
        if not torch.isfinite(loss):
            print(f"[ERROR][EVAL] Non-finite loss at batch {batch_idx}: loss={loss.item() if loss.numel()==1 else loss}, paths[0]={paths[0] if isinstance(paths, (list, tuple)) and len(paths)>0 else paths}")

        geo_err = batch_geo_distance_km(output, target, paths)
        metric_logger.update(loss=loss.item(), geo_km=geo_err)

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
