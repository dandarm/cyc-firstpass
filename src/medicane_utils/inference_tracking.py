"""Inference entry point for cyclone centre tracking."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import warnings

import engine_for_tracking as tracking_engine
import utils
from arguments import prepare_tracking_args
from dataset.data_manager import DataManager
from models.tracking_model import create_tracking_model
from utils import setup_for_distributed

warnings.filterwarnings("ignore")
utils.suppress_transformers_pytree_warning()


def _format_log_value(key: str, value):
    """Format numeric stats preserving LR precision."""
    if not isinstance(value, float):
        return value
    if "lr" in key.lower():
        return float(f"{value:.8g}")
    return round(value, 4)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True


def _ensure_path_list(paths) -> List[str]:
    if isinstance(paths, (list, tuple)):
        return [str(p) for p in paths]
    return [str(paths)]


def _compute_sample_record(path: str, pred_xy: torch.Tensor, target_xy: torch.Tensor) -> Dict[str, float | str | None]:
    pred_np = pred_xy.detach().cpu().numpy().astype(float)
    target_np = target_xy.detach().cpu().numpy().astype(float)
    err_vec = pred_np - target_np
    err_px = float(np.linalg.norm(err_vec))

    offset_x = offset_y = None
    pred_abs = target_abs = None
    pred_lat = pred_lon = target_lat = target_lon = None
    err_km = None

    try:
        offset_x, offset_y = tracking_engine._parse_tile_offsets(path)
        pred_abs = pred_np + np.array([offset_x, offset_y], dtype=float)
        target_abs = target_np + np.array([offset_x, offset_y], dtype=float)
        lat_pred, lon_pred = tracking_engine._pixels_to_latlon(
            np.array([pred_abs[0]]), np.array([pred_abs[1]])
        )
        lat_true, lon_true = tracking_engine._pixels_to_latlon(
            np.array([target_abs[0]]), np.array([target_abs[1]])
        )
        pred_lat = float(lat_pred[0])
        pred_lon = float(lon_pred[0])
        target_lat = float(lat_true[0])
        target_lon = float(lon_true[0])
        err_km = float(
            tracking_engine._haversine_km(lat_pred, lon_pred, lat_true, lon_true)[0]
        )
    except Exception as exc:  # pragma: no cover - defensive path for malformed names
        print(f"[WARN][tracking_inference] Unable to recover geo coords for {path}: {exc}")

    record = {
        "path": path,
        "pred_x": float(pred_np[0]),
        "pred_y": float(pred_np[1]),
        "target_x": float(target_np[0]),
        "target_y": float(target_np[1]),
        "err_px": err_px,
        "err_km": err_km,
    }

    if offset_x is not None and pred_abs is not None and target_abs is not None:
        record.update(
            {
                "tile_offset_x": float(offset_x),
                "tile_offset_y": float(offset_y),
                "pred_x_global": float(pred_abs[0]),
                "pred_y_global": float(pred_abs[1]),
                "target_x_global": float(target_abs[0]),
                "target_y_global": float(target_abs[1]),
                "pred_lat": pred_lat,
                "pred_lon": pred_lon,
                "target_lat": target_lat,
                "target_lon": target_lon,
            }
        )

    return record


@torch.no_grad()
def inference_epoch(model, data_loader, device) -> tuple[Dict[str, float], List[Dict[str, float | str | None]]]:
    criterion = torch.nn.MSELoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    results: List[Dict[str, float | str | None]] = []

    for samples, target, paths in metric_logger.log_every(data_loader, 20, header="Infer:"):
        samples = samples.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(samples)
        loss = criterion(output, target)

        geo_err = tracking_engine.batch_geo_distance_km(output, target, _ensure_path_list(paths))
        px_err = float(torch.norm(output - target, dim=-1).mean().item())

        metric_logger.update(loss=loss.item(), geo_km=geo_err, px_err=px_err)

        path_list = _ensure_path_list(paths)
        for idx, sample_path in enumerate(path_list):
            results.append(
                _compute_sample_record(sample_path, output[idx], target[idx])
            )

    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats, results


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    if not checkpoint_path:
        raise ValueError("checkpoint path must be provided for inference")
    map_location = device if device.type == "cpu" else torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("model") or checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} does not contain 'model' or 'state_dict' keys; "
            f"available keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'}"
        )
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Loaded checkpoint from {checkpoint_path}: {msg}")


def launch_inference_tracking(terminal_args: argparse.Namespace) -> None:
    args = prepare_tracking_args(machine=terminal_args.on)

    if terminal_args.csvfile:
        args.test_path = terminal_args.csvfile

    args.init_ckpt = terminal_args.inference_model
    args.load_for_test_mode = True
    args.distributed = False

    set_seeds(args.seed)

    rank, local_rank, world_size, _, _ = utils.get_resources()

    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        args.distributed = True
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        args.gpu = local_rank
        args.world_size = world_size
        args.rank = rank
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        args.gpu = None
        args.world_size = 1
        args.rank = 0

    if args.log_dir and not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    setup_for_distributed(rank == 0)

    data_manager = DataManager(
        is_train=False,
        args=args,
        type_t="supervised",
        world_size=world_size,
        rank=rank,
        specify_data_path=args.test_path,
    )
    data_loader = data_manager.get_tracking_dataloader(args)

    model = create_tracking_model(args.model, **args.__dict__)
    model.to(device)
    load_checkpoint(model, args.init_ckpt, device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu
        )

    model.eval()
    torch.cuda.empty_cache()

    start = time.time()
    stats, local_results = inference_epoch(model, data_loader, device)

    gathered_results = local_results
    if dist.is_available() and dist.is_initialized():
        gathered: List[List[Dict[str, float | str | None]]] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_results)
        if utils.is_main_process():
            gathered_results = [item for sublist in gathered for item in sublist]
        else:
            gathered_results = []

    if utils.is_main_process():
        df = pd.DataFrame(gathered_results)
        preds_csv = terminal_args.preds_csv
        out_csv = preds_csv if not args.output_dir else os.path.join(args.output_dir, preds_csv)
        try:
            df.to_csv(out_csv, index=False)
            print(f"Saved tracking predictions to {out_csv}")
        except Exception as exc:
            print(f"Warning: could not save predictions CSV: {exc}")

        log_stats = {f"test_{k}": v for k, v in stats.items()}
        log_stats = {k: _format_log_value(k, v) for k, v in log_stats.items()}
        metrics_path = os.path.join(args.output_dir, "inference_tracking_metrics.txt")
        try:
            with open(metrics_path, "a") as handle:
                handle.write(json.dumps(log_stats) + "\n")
        except Exception as exc:
            print(f"Warning: could not write metrics to {metrics_path}: {exc}")

    dist.barrier() if dist.is_available() and dist.is_initialized() else None

    elapsed = time.time() - start
    if utils.is_main_process():
        print(f"Inference time: {datetime.timedelta(seconds=int(elapsed))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Cyclone tracking inference", add_help=False)
    parser.add_argument("--on", type=str, default="leonardo", help="[ewc, leonardo]")
    parser.add_argument("--csvfile", type=str, default="val_tracking.csv", help="CSV con clip di tracking")
    parser.add_argument(
        "--inference_model",
        type=str,
        default="output/checkpoint-tracking-best.pth",
        help="Checkpoint da utilizzare per l'inferenza",
    )
    parser.add_argument(
        "--preds_csv",
        type=str,
        default="tracking_inference_predictions.csv",
        help="Nome del CSV con le predizioni salvate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    launch_inference_tracking(cli_args)
