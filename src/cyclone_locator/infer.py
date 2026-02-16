import argparse
import math
import json
import logging
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

from cyclone_locator.datasets.temporal_utils import TemporalWindowSelector
from cyclone_locator.models.simplebaseline import SimpleBaseline
from cyclone_locator.models.x3d_backbone import X3DBackbone
from cyclone_locator.models.energy_fusion import EnergyFusion, compute_energy_features
from cyclone_locator.utils.geometry import crop_square
from cyclone_locator.utils.metric import peak_and_width
from cyclone_locator import metrics as metrics_lib
from cyclone_locator import geo_utils


LOGGER_NAME = "cyclone_eval"
PATH_ALIASES = ["image_path", "path", "resized_path"]
PRESENCE_ALIASES = ["presence"]
X_ALIASES = ["x_pix_resized", "x_g", "x", "x_center"]
Y_ALIASES = ["y_pix_resized", "y_g", "y", "y_center"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Cyclone first-pass inference/eval")
    parser.add_argument("--config", default="config/default.yml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--letterbox-meta", default=None,
                        help="CSV con scale/padding per la back-projection (richiesto se --export-roi)")
    parser.add_argument("--export-roi", action="store_true",
                        help="Abilita il salvataggio dei ritagli ROI + coordinate originali")
    parser.add_argument("--roi-dir", default=None,
                        help="Directory dove salvare i ritagli ROI (default: <out_dir>/roi)")
    parser.add_argument("--out_dir", default=None,
                        help="Directory base per output (default: <checkpoint_dir>/eval)")
    parser.add_argument("--save-preds", default=None,
                        help="Percorso CSV per salvare le predizioni (default: <out_dir>/preds.csv)")
    parser.add_argument("--metrics-out", default=None,
                        help="Percorso JSON per salvare le metriche aggregate (opzionale)")
    parser.add_argument("--sweep-curves", default=None,
                        help="Directory dove scrivere PR/ROC curve (opzionale)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Soglia presence τ per metriche/ROI")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--bs", type=int, default=None,
                        help="Alias di --batch-size (come train.py)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None,
                        help="cuda|cpu (default: cuda se disponibile altrimenti cpu)")
    parser.add_argument("--amp", action="store_true",
                        help="Abilita autocast AMP in inferenza")
    parser.add_argument("--soft-argmax", action="store_true",
                        help="Usa soft-argmax per decodificare il centro (default argmax)")
    parser.add_argument("--soft-argmax-tau", type=float, default=None,
                        help="Temperatura τ per soft-argmax/DSNT (più alta -> più smooth)")
    parser.add_argument("--oracle-localization", action="store_true",
                        help="Valuta l'errore centro su tutti i GT positivi (ignora decisione binaria)")
    parser.add_argument("--center-thresholds-px", type=float, nargs="+", default=[8, 16, 24, 32])
    parser.add_argument("--center-thresholds-km", type=float, nargs="+", default=[25, 50, 100, 200])
    parser.add_argument("--heatmap-stride", type=int, default=None,
                        help="Override stride della heatmap (default: config.train.heatmap_stride)")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Override lato immagine (default: config.train.image_size)")
    parser.add_argument("--roi-base-radius", type=int, default=None,
                        help="Override raggio base ROI in px")
    parser.add_argument("--roi-sigma-multiplier", type=float, default=None,
                        help="Override moltiplicatore sigma ROI")
    parser.add_argument("--temporal_T", type=int, default=None,
                        help="Override dimensione finestra temporale (default: config.train.temporal_T o 1)")
    parser.add_argument("--temporal_stride", type=int, default=None,
                        help="Override stride temporale tra frame (default: config.train.temporal_stride o 1)")
    parser.add_argument("--backbone", default=None, help="Override del backbone (default: config.train.backbone)")
    parser.add_argument("--presence-from-peak", action="store_true",
                        help="Usa solo il picco della heatmap come presenza (ignora la head presence)")
    parser.add_argument("--presence-mode", choices=["head", "peak", "both", "energy"], default=None,
                        help="Presence mode (override, default: infer.presence_mode/train.presence_mode)")
    parser.add_argument("--peak-mode", choices=["logsumexp", "energy"], default=None,
                        help="Peak mode override (default: loss.peak_mode)")
    parser.add_argument("--peak-threshold", type=float, default=None,
                        help="Soglia τ da usare quando presence-from-peak è attivo (default: --threshold o infer.presence_threshold)")
    parser.add_argument("--peak-pool", choices=["max", "logsumexp"], default=None,
                        help="Pooling per presence-from-peak (default: infer.peak_pool o 'max')")
    parser.add_argument("--peak-tau", type=float, default=None,
                        help="Temperatura τ per peak-pool=logsumexp (default: infer.peak_tau o 1.0)")
    parser.add_argument("--presence-topk", type=int, default=None,
                        help="Top-K per presence-from-peak con logsumexp (default: train.presence_topk o 0=all)")
    
    return parser.parse_args()


def setup_logging(out_dir: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(os.path.join(out_dir, "eval.log"))
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def resolve_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for alias in aliases:
        cand = lower_map.get(alias.lower(), None)
        if cand is not None:
            return cand
    return None


def normalize_path(p: str, base_dir: str) -> str:
    p = str(p).strip()
    if not p:
        return p
    if os.path.isabs(p):
        return os.path.abspath(p)
    return os.path.abspath(os.path.join(base_dir, p))


def load_manifest(manifest_csv: str, logger: logging.Logger) -> pd.DataFrame:
    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"manifest_csv not found: {manifest_csv}")
    df = pd.read_csv(manifest_csv)
    manifest_dir = os.path.dirname(os.path.abspath(manifest_csv))

    rename_log: List[str] = []
    path_col = resolve_column(df, PATH_ALIASES)
    if path_col is None:
        raise ValueError(f"manifest missing any path column from {PATH_ALIASES}")
    if path_col != "image_path":
        rename_log.append(f"{path_col}->image_path")
        df = df.rename(columns={path_col: "image_path"})

    presence_col = resolve_column(df, PRESENCE_ALIASES)
    if presence_col and presence_col != "presence":
        rename_log.append(f"{presence_col}->presence")
        df = df.rename(columns={presence_col: "presence"})

    x_col = resolve_column(df, X_ALIASES)
    if x_col and x_col != "x_pix_resized":
        rename_log.append(f"{x_col}->x_pix_resized")
        df = df.rename(columns={x_col: "x_pix_resized"})
    y_col = resolve_column(df, Y_ALIASES)
    if y_col and y_col != "y_pix_resized":
        rename_log.append(f"{y_col}->y_pix_resized")
        df = df.rename(columns={y_col: "y_pix_resized"})

    if rename_log:
        logger.info("Manifest columns normalized: %s", ", ".join(rename_log))

    df["image_path"] = df["image_path"].astype(str).apply(lambda p: normalize_path(p, manifest_dir))
    missing_path_mask = df["image_path"] == ""
    if missing_path_mask.any():
        logger.warning("Dropping %d rows with empty image_path", missing_path_mask.sum())
        df = df[~missing_path_mask]

    exists_mask = df["image_path"].apply(os.path.exists)
    missing = df[~exists_mask]
    if not missing.empty:
        logger.warning("Skipping %d/%d files missing on disk (first: %s)",
                       len(missing), len(df), missing["image_path"].iloc[0])
        df = df[exists_mask]
    df = df.reset_index(drop=True)

    if "presence" in df.columns:
        df["presence"] = pd.to_numeric(df["presence"], errors="coerce")
    if "x_pix_resized" in df.columns:
        df["x_pix_resized"] = pd.to_numeric(df["x_pix_resized"], errors="coerce")
    if "y_pix_resized" in df.columns:
        df["y_pix_resized"] = pd.to_numeric(df["y_pix_resized"], errors="coerce")
    if "cx" in df.columns:
        df["cx"] = pd.to_numeric(df["cx"], errors="coerce")
    if "cy" in df.columns:
        df["cy"] = pd.to_numeric(df["cy"], errors="coerce")

    if "datetime" in df.columns:
        df["_sort_dt"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values(by=["_sort_dt", "image_path"]).drop(columns=["_sort_dt"])
    else:
        df = df.sort_values(by="image_path")
    df = df.reset_index(drop=True)
    df["manifest_idx"] = np.arange(len(df))

    total = len(df)
    n_presence = int(df["presence"].notna().sum()) if "presence" in df.columns else 0
    n_center = int(
        (df.get("x_pix_resized").notna() & df.get("y_pix_resized").notna()).sum()
    ) if {"x_pix_resized", "y_pix_resized"}.issubset(df.columns) else 0
    logger.info("Manifest loaded: %d records (%d with presence GT, %d with center GT)",
                total, n_presence, n_center)
    return df


def load_letterbox_meta(meta_csv: str) -> Dict[str, Dict[str, float]]:
    required_base = {"orig_path", "resized_path", "pad_x", "pad_y", "orig_w", "orig_h", "out_size"}
    meta_df = pd.read_csv(meta_csv)
    before = len(meta_df)
    meta_df = meta_df.dropna(subset=list(required_base))
    if len(meta_df) < before:
        logging.getLogger(LOGGER_NAME).warning("Dropped %d rows with NaN in letterbox meta %s", before - len(meta_df), meta_csv)
    cols = set(meta_df.columns)
    missing = required_base - cols
    if missing:
        raise ValueError(f"letterbox meta missing columns: {sorted(missing)}")
    has_scale = "scale" in cols
    has_scale_xy = {"scale_x", "scale_y"}.issubset(cols)
    if not (has_scale or has_scale_xy):
        raise ValueError("letterbox meta must include either 'scale' or ('scale_x' and 'scale_y')")
    base_dir = os.path.dirname(os.path.abspath(meta_csv))
    meta_map: Dict[str, Dict[str, float]] = {}
    def _add_key(meta_map: Dict[str, Dict[str, float]], key: str, entry: Dict[str, float]) -> None:
        if not key:
            return
        meta_map[key] = entry
        # Also index by realpath to avoid mismatches with symlinks/alternate mounts.
        real_key = os.path.realpath(key)
        if real_key:
            meta_map[real_key] = entry

    for _, row in meta_df.iterrows():
        scale_x = float(row["scale_x"]) if "scale_x" in row else float(row["scale"])
        scale_y = float(row["scale_y"]) if "scale_y" in row else float(row["scale"])
        scale = float(row["scale"]) if "scale" in row and np.isfinite(row["scale"]) else float(scale_x)
        entry = {
            "orig_path": normalize_path(row["orig_path"], base_dir),
            "resized_path": normalize_path(row["resized_path"], base_dir),
            "scale": float(scale),  # legacy scalar (use scale_x/scale_y for non-uniform resizes)
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "pad_x": float(row["pad_x"]),
            "pad_y": float(row["pad_y"]),
            "orig_w": int(row["orig_w"]),
            "orig_h": int(row["orig_h"]),
            "out_size": int(row["out_size"])
        }
        _add_key(meta_map, entry["orig_path"], entry)
        _add_key(meta_map, entry["resized_path"], entry)
    return meta_map


def _convert_letterbox_to_original(x_lb: float, y_lb: float, meta: Dict[str, float]) -> Tuple[float, float]:
    sx = float(meta.get("scale_x", meta["scale"]))
    sy = float(meta.get("scale_y", meta["scale"]))
    x_orig = (x_lb - meta["pad_x"]) / sx
    y_orig = (y_lb - meta["pad_y"]) / sy
    return float(x_orig), float(y_orig)


def attach_original_xy(
    df: pd.DataFrame,
    meta_map: Dict[str, Dict[str, float]],
    x_col: str,
    y_col: str,
    out_x: str,
    out_y: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    x_list: List[float] = []
    y_list: List[float] = []
    missing_meta = 0
    for row in df.itertuples():
        x_val = getattr(row, x_col, np.nan)
        y_val = getattr(row, y_col, np.nan)
        if not (np.isfinite(x_val) and np.isfinite(y_val)):
            x_list.append(np.nan)
            y_list.append(np.nan)
            continue
        meta = meta_map.get(row.image_path)
        if meta is None:
            missing_meta += 1
            x_list.append(np.nan)
            y_list.append(np.nan)
            continue
        x_orig, y_orig = _convert_letterbox_to_original(x_val, y_val, meta)
        x_list.append(x_orig)
        y_list.append(y_orig)
    if missing_meta:
        logger.warning("Missing letterbox meta for %d samples while computing %s/%s", missing_meta, out_x, out_y)
    df[out_x] = x_list
    df[out_y] = y_list
    return df


def attach_manifest_gt_orig(manifest_df: pd.DataFrame, meta_map: Optional[Dict[str, Dict[str, float]]],
                            logger: logging.Logger) -> pd.DataFrame:
    if "x_orig_gt" in manifest_df.columns and "y_orig_gt" in manifest_df.columns:
        return manifest_df
    if {"cx", "cy"}.issubset(manifest_df.columns):
        manifest_df["x_orig_gt"] = pd.to_numeric(manifest_df["cx"], errors="coerce")
        manifest_df["y_orig_gt"] = pd.to_numeric(manifest_df["cy"], errors="coerce")
        return manifest_df
    if meta_map is not None and {"x_pix_resized", "y_pix_resized"}.issubset(manifest_df.columns):
        manifest_df = attach_original_xy(
            manifest_df,
            meta_map,
            x_col="x_pix_resized",
            y_col="y_pix_resized",
            out_x="x_orig_gt",
            out_y="y_orig_gt",
            logger=logger,
        )
        return manifest_df
    logger.warning("Manifest lacks cx/cy and cannot map resized keypoints to original coordinates; "
                   "geo metrics in km will be unavailable.")
    return manifest_df


def compute_geo_center_metrics(
    joined: pd.DataFrame,
    eval_mask: np.ndarray,
    thresholds_km: Optional[Sequence[float]],
    localization_policy: str,
    logger: logging.Logger,
) -> Optional[Dict[str, object]]:
    required_cols = ["x_orig", "y_orig", "x_orig_gt", "y_orig_gt"]
    if not set(required_cols).issubset(joined.columns):
        logger.info("Skipping geo metrics: missing columns %s", required_cols)
        return None
    coords = joined[required_cols].to_numpy(dtype=float)
    valid = eval_mask & np.isfinite(coords).all(axis=1)
    if not valid.any():
        logger.info("Skipping geo metrics: no valid samples with original coordinates.")
        return None
    pred_x = coords[valid, 0]
    pred_y = coords[valid, 1]
    gt_x = coords[valid, 2]
    gt_y = coords[valid, 3]
    try:
        lat_pred, lon_pred = geo_utils.pixels_to_latlon(pred_x, pred_y)
        lat_gt, lon_gt = geo_utils.pixels_to_latlon(gt_x, gt_y)
        distances = geo_utils.haversine_km(lat_pred, lon_pred, lat_gt, lon_gt)
    except Exception as exc:  # pragma: no cover - runtime dependency (Basemap) missing
        logger.warning("Impossibile calcolare metriche km: %s", exc)
        return None
    finite = np.isfinite(distances)
    if not finite.any():
        logger.info("Skipping geo metrics: haversine distances all invalid.")
        return None
    distances = distances[finite]
    logger.info("Geo metrics computed on %d samples", len(distances))
    return metrics_lib.center_metrics_summary(distances, thresholds=thresholds_km, unit="km", policy=localization_policy)


class EvalDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, image_size: int, logger: logging.Logger,
                 temporal_T: int = 1, temporal_stride: int = 1):
        self.df = manifest_df.reset_index(drop=True)
        self.logger = logger
        self.image_size = int(image_size)
        self.warned_size = False
        self.temporal_selector = TemporalWindowSelector(temporal_T, temporal_stride)

    def __len__(self) -> int:
        return len(self.df)

    def _normalize_frame(self, frame_np: np.ndarray) -> np.ndarray:
        if frame_np.ndim == 2:
            frame_np = frame_np[..., None]
        frame_np = frame_np.astype(np.float32)
        return frame_np / 255.0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        path = row["image_path"]

        center_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if center_img is None:
            raise FileNotFoundError(f"cannot read image: {path}")
        h, w = center_img.shape[:2]
        if (h != self.image_size or w != self.image_size) and not self.warned_size:
            self.logger.warning("Image size mismatch (%s vs %d). Continuing anyway.", (h, w), self.image_size)
            self.warned_size = True

        window_paths = self.temporal_selector.get_window(path)
        frames = []
        for p in window_paths:
            if p == path:
                img_np = center_img
            else:
                try:
                    img_np = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                except Exception:
                    img_np = None
                if img_np is None:
                    img_np = center_img
            if img_np.shape != center_img.shape:
                img_np = center_img
            frames.append(img_np)

        frames = [self._normalize_frame(f) for f in frames]
        ch = frames[0].shape[2]
        fused = np.concatenate(frames, axis=2)
        if fused.shape[2] != ch * len(frames) and not self.warned_size:
            self.logger.warning("Channel mismatch in temporal fusion (%d vs expected %d)", fused.shape[2], ch * len(frames))
            self.warned_size = True
        tensor = torch.from_numpy(fused).permute(2, 0, 1)
        video = torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in frames], dim=0).permute(1, 0, 2, 3)
        return {
            "image": tensor,
            "video": video,
            "image_path": path,
            "manifest_idx": int(row["manifest_idx"])
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    videos = torch.stack([b["video"] for b in batch], dim=0)
    paths = [b["image_path"] for b in batch]
    manifest_idx = [b["manifest_idx"] for b in batch]
    return {"image": images, "video": videos, "image_path": paths, "manifest_idx": manifest_idx}


def build_model(
    cfg: dict,
    checkpoint_path: str,
    device: torch.device,
    logger: logging.Logger,
    temporal_T: int,
    heatmap_stride: int,
    use_energy_fusion: bool = False,
) -> torch.nn.Module:
    backbone = cfg.get("train", {}).get("backbone", "resnet18")
    pretrained = bool(cfg.get("train", {}).get("backbone_pretrained", True))
    if backbone.startswith("x3d"):
        model = X3DBackbone(backbone=backbone, pretrained=pretrained, heatmap_stride=int(heatmap_stride))
    else:
        model = SimpleBaseline(backbone=backbone, temporal_T=temporal_T, pretrained=pretrained, heatmap_stride=int(heatmap_stride))
    if use_energy_fusion:
        loss_cfg = cfg.get("loss", {})
        energy_b0 = float(loss_cfg.get("energy_init_b0", 0.0) or 0.0)
        energy_wE = float(loss_cfg.get("energy_init_wE", 1.0) or 1.0)
        energy_wC = float(loss_cfg.get("energy_init_wC", 1.0) or 1.0)
        energy_wH = float(loss_cfg.get("energy_init_wH", 1.0) or 1.0)
        model.energy_fusion = EnergyFusion(b0=energy_b0, wE=energy_wE, wC=energy_wC, wH=energy_wH)
    state = torch.load(checkpoint_path, map_location="cpu")
    weights = state.get("model", state)
    if not use_energy_fusion:
        drop_keys = [k for k in weights.keys() if str(k).startswith("energy_fusion.")]
        if drop_keys:
            for k in drop_keys:
                weights.pop(k, None)
            logger.warning("Dropped %d energy_fusion params from checkpoint (use_energy_fusion=False).", len(drop_keys))
    try:
        model.load_state_dict(weights, strict=True)
    except RuntimeError as exc:
        if use_energy_fusion:
            logger.warning("Energy fusion params missing in checkpoint; loading with strict=False. (%s)", exc)
            model.load_state_dict(weights, strict=False)
        else:
            raise
    model.to(device)
    model.eval()
    logger.info("Loaded checkpoint %s with backbone=%s", checkpoint_path, backbone)
    return model


def decode_heatmap(hm: np.ndarray, stride: int, soft: bool = False, tau: float = 1.0) -> Tuple[float, float]:
    if soft:
        if tau <= 0:
            raise ValueError("tau must be > 0")
        logits = (hm - hm.max()) / float(tau)
        weights = np.exp(logits)
        total = weights.sum()
        if total == 0:
            return 0.0, 0.0
        weights /= total
        yy, xx = np.mgrid[0:hm.shape[0], 0:hm.shape[1]]
        x = float((weights * xx).sum())
        y = float((weights * yy).sum())
    else:
        idx = np.argmax(hm)
        h, w = hm.shape
        y = idx // w
        x = idx % w
    return x * stride, y * stride


def combine_presence(presence_prob: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
    """
    presence_prob: (B,) con sigmoid(logits)
    heatmap: (B,H,W) o (B,1,H,W)
    Restituisce una media semplice tra la presenza predetta e il picco della heatmap.
    """
    if heatmap.ndim == 4 and heatmap.shape[1] == 1:
        heatmap = heatmap.squeeze(1)
    peak = heatmap.amax(dim=[-1, -2])
    return 0.5 * presence_prob + 0.5 * peak


def spatial_peak_pool(
    heatmap_logits: torch.Tensor,
    pool: str = "max",
    tau: float = 1.0,
    topk: int | None = None,
) -> torch.Tensor:
    """
    heatmap_logits: (B,H,W) or (B,1,H,W)
    Returns: (B,) pooled logits.
    """
    if heatmap_logits.ndim == 4 and heatmap_logits.shape[1] == 1:
        x = heatmap_logits.squeeze(1)
    elif heatmap_logits.ndim == 3:
        x = heatmap_logits
    else:
        raise ValueError(f"Expected heatmap logits shape (B,H,W) or (B,1,H,W), got {heatmap_logits.shape}")
    pool = str(pool).lower().strip()
    if pool == "max":
        return x.amax(dim=[-1, -2])
    if pool == "logsumexp":
        if tau <= 0:
            raise ValueError("tau must be > 0 for logsumexp pooling")
        # Do pooling in float32 for stability under AMP / small tau.
        x32 = (x / float(tau)).float()
        if topk is not None and int(topk) > 0:
            k = min(int(topk), x32.shape[-1] * x32.shape[-2])
            flat = x32.flatten(1)
            vals, _ = torch.topk(flat, k=k, dim=1)
            pooled = torch.logsumexp(vals, dim=1)
        else:
            pooled = torch.logsumexp(x32, dim=[-1, -2])
        return float(tau) * pooled
    raise ValueError(f"Unknown peak_pool: {pool}")


def _recenter_peak_logit(
    peak_logit: torch.Tensor,
    logits: torch.Tensor,
    topk: int | None,
    mode: str,
) -> torch.Tensor:
    mode = str(mode or "none").lower().strip()
    if peak_logit.ndim == 1:
        peak_logit = peak_logit.unsqueeze(1)
    if mode in {"none", ""}:
        return peak_logit
    if mode in {"logk", "logk+median", "median+logk"}:
        k = int(topk or (logits.shape[-1] * logits.shape[-2]))
        if k > 0:
            peak_logit = peak_logit - math.log(k)
    if mode in {"median", "logk+median", "median+logk"}:
        flat = logits.squeeze(1).flatten(1)
        med = flat.median(dim=1).values
        peak_logit = peak_logit - med.unsqueeze(1)
    return peak_logit


def compute_peak_logit(
    logits: torch.Tensor,
    pool: str,
    tau: float,
    topk: int | None,
    center_mode: str,
) -> torch.Tensor:
    peak_logit = spatial_peak_pool(logits, pool=pool, tau=tau, topk=topk)
    if peak_logit.ndim == 1:
        peak_logit = peak_logit.unsqueeze(1)
    if str(pool).lower().strip() == "logsumexp":
        peak_logit = _recenter_peak_logit(peak_logit, logits, topk, center_mode)
    return peak_logit


def run_inference(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    stride: int,
    soft_argmax: bool,
    soft_argmax_tau: float,
    amp: bool,
    presence_mode: str = "head",
    peak_pool: str = "max",
    peak_tau: float = 1.0,
    presence_topk: int | None = None,
    peak_logit_alpha: float = 0.5,
    peak_logit_center: str = "none",
    peak_mode: str = "logsumexp",
    dsnt_tau: float = 1.0,
) -> List[Dict[str, float]]:
    predictions: List[Dict[str, float]] = []
    autocast_enabled = amp and device.type == "cuda"
    start = time.time()
    total = 0
    checked_stride = False
    with torch.no_grad():
        for batch in data_loader:
            input_key = "video" if getattr(model, "input_is_video", False) else "image"
            images = batch[input_key].to(device)
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                heatmaps_pred, logits = model(images)
            if not checked_stride:
                frame_h = int(images.shape[-2])
                hm_h = int(heatmaps_pred.shape[-2])
                if hm_h <= 0:
                    raise ValueError(f"Invalid heatmap height: {hm_h}")
                stride_real = frame_h / float(hm_h)
                if abs(stride_real - float(stride)) > 1e-6:
                    raise ValueError(
                        f"heatmap_stride mismatch: configured stride={stride} but model output implies stride={stride_real:.4f} "
                        f"(frame_h={frame_h}, hm_h={hm_h}). "
                        "Fix: pass the correct `--heatmap-stride` (or use a checkpoint/config with matching stride)."
                    )
                checked_stride = True
            heatmaps = heatmaps_pred.squeeze(1)
            probs_raw = torch.sigmoid(logits).squeeze(1)
            presence_mode = str(presence_mode or "head").lower().strip()
            if presence_mode not in {"head", "peak", "both", "energy"}:
                raise ValueError(f"Invalid presence_mode: {presence_mode}")
            peak_mode = str(peak_mode or "logsumexp").lower().strip()
            use_energy = presence_mode == "energy"
            use_peak = presence_mode in {"peak", "both"}
            if use_peak and peak_mode == "energy":
                logging.getLogger(LOGGER_NAME).warning(
                    "peak_mode=energy ignored because presence_mode is not energy; falling back to logsumexp."
                )
                peak_mode = "logsumexp"
            peak_logit = None
            if use_energy:
                energy_fusion = getattr(model, "energy_fusion", None)
                if energy_fusion is None:
                    raise ValueError("energy_fusion module not found on model (presence_mode=energy)")
                E, C = compute_energy_features(heatmaps_pred, dsnt_tau=float(dsnt_tau), topk=presence_topk)
                peak_logit = energy_fusion(E, C, logits)
            elif use_peak:
                peak_logit = compute_peak_logit(
                    heatmaps_pred,
                    pool=peak_pool,
                    tau=float(peak_tau),
                    topk=presence_topk,
                    center_mode=peak_logit_center,
                )
            if presence_mode == "both" and peak_logit is not None:
                comb_logit = logits + float(peak_logit_alpha) * peak_logit
                combined_probs = torch.sigmoid(comb_logit).squeeze(1)
            elif use_energy and peak_logit is not None:
                combined_probs = torch.sigmoid(peak_logit).squeeze(1)
            elif use_peak and peak_logit is not None:
                combined_probs = torch.sigmoid(peak_logit).squeeze(1)
            else:
                combined_probs = probs_raw
            combined_probs = torch.clamp(combined_probs, 1e-6, 1 - 1e-6)

            heatmaps_np = heatmaps.cpu().numpy()
            probs_np = combined_probs.cpu().numpy()
            probs_raw_np = probs_raw.cpu().numpy()
            peak_prob = torch.sigmoid(peak_logit).squeeze(1) if peak_logit is not None else torch.sigmoid(logits).squeeze(1)
            peaks_np = peak_prob.detach().cpu().numpy()
            logits_np = logits.squeeze(1).cpu().numpy()
            manifest_idx_batch = batch["manifest_idx"]
            for i, path in enumerate(batch["image_path"]):
                hm = heatmaps_np[i]
                x_g, y_g = decode_heatmap(hm, stride=stride, soft=soft_argmax, tau=float(soft_argmax_tau))
                _, _, _, width = peak_and_width(hm)
                predictions.append({
                    "image_path": path,
                    "manifest_idx": int(manifest_idx_batch[i]),
                    "presence_prob": float(probs_np[i]),
                    "presence_prob_raw": float(probs_raw_np[i]),
                    "presence_logit": float(logits_np[i]),
                    "heatmap_peak": float(peaks_np[i]),
                    "x_g": float(x_g),
                    "y_g": float(y_g),
                    "peak_width_heatmap": float(width),
                })
            total += len(images)
    elapsed = time.time() - start
    ips = total / elapsed if elapsed > 0 else 0.0
    time_per_1k = (elapsed / total * 1000.0) if total else float("nan")
    logging.getLogger(LOGGER_NAME).info("Inference done on %d samples in %.1fs (%.1f img/s, %.2fs / 1k)",
                                        total, elapsed, ips, time_per_1k)
    return predictions


def apply_backprojection_and_roi(
    preds_df: pd.DataFrame,
    meta_map: Dict[str, Dict[str, float]],
    roi_dir: str,
    threshold: float,
    roi_base_radius: int,
    roi_sigma_multiplier: float,
    stride: int,
) -> pd.DataFrame:
    os.makedirs(roi_dir, exist_ok=True)
    roi_paths: List[str] = []
    roi_radius: List[float] = []
    missing_meta = 0
    saved = 0
    for row in preds_df.itertuples():
        meta = meta_map.get(row.image_path)
        if meta is None:
            missing_meta += 1
            roi_paths.append("")
            roi_radius.append(np.nan)
            continue
        x_orig = getattr(row, "x_orig", np.nan)
        y_orig = getattr(row, "y_orig", np.nan)
        if not np.isfinite(x_orig) or not np.isfinite(y_orig):
            roi_paths.append("")
            roi_radius.append(np.nan)
            continue
        x_orig_clipped = float(np.clip(x_orig, 0, max(meta["orig_w"] - 1, 0)))
        y_orig_clipped = float(np.clip(y_orig, 0, max(meta["orig_h"] - 1, 0)))
        r_dynamic = roi_sigma_multiplier * row.peak_width_heatmap * stride if row.peak_width_heatmap > 0 else 0
        r_crop = max(roi_base_radius, int(round(r_dynamic)))
        if row.presence_prob >= threshold:
            roi_radius.append(r_crop)
            orig_img = cv2.imread(meta["orig_path"], cv2.IMREAD_UNCHANGED)
            if orig_img is None:
                roi_paths.append("")
                continue
            roi = crop_square(orig_img, (x_orig_clipped, y_orig_clipped), r_crop)
            stem = os.path.splitext(os.path.basename(meta["orig_path"]))[0]
            roi_name = f"{row.manifest_idx:06d}_{stem}.png"
            roi_path = os.path.join(roi_dir, roi_name)
            cv2.imwrite(roi_path, roi)
            roi_paths.append(roi_path)
            saved += 1
        else:
            roi_paths.append("")
            roi_radius.append(np.nan)
    logging.getLogger(LOGGER_NAME).info("Back-projection complete. ROI salvate: %d (missing meta: %d)",
                                        saved, missing_meta)
    preds_df["roi_path"] = roi_paths
    preds_df["roi_radius_px"] = roi_radius
    return preds_df


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    default_out_dir = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "eval")
    out_dir = args.out_dir or default_out_dir
    logger = setup_logging(out_dir)
    logger.info("Starting inference")
    logger.info("Args: %s", vars(args))

    image_size = args.image_size or cfg.get("train", {}).get("image_size", 512)
    stride = args.heatmap_stride or cfg.get("train", {}).get("heatmap_stride", 4)
    temporal_T = max(1, int(args.temporal_T or cfg.get("train", {}).get("temporal_T", 1)))
    temporal_stride = max(1, int(args.temporal_stride or cfg.get("train", {}).get("temporal_stride", 1)))
    batch_size = args.bs or args.batch_size or cfg.get("train", {}).get("batch_size", 32)
    presence_threshold_default = cfg.get("infer", {}).get("presence_threshold", 0.5)
    presence_mode_cfg = cfg.get("infer", {}).get("presence_mode") or cfg.get("train", {}).get("presence_mode")
    if args.presence_mode:
        presence_mode = str(args.presence_mode)
    elif presence_mode_cfg:
        presence_mode = str(presence_mode_cfg)
    else:
        presence_mode = "peak" if bool(cfg.get("infer", {}).get("presence_from_peak", False)) else "head"
    presence_mode = str(presence_mode or "head").lower().strip()
    if presence_mode not in {"head", "peak", "both", "energy"}:
        raise ValueError(f"Invalid presence_mode: {presence_mode}")
    presence_from_peak = presence_mode == "peak"
    peak_pool_default = cfg.get("infer", {}).get("peak_pool", "max")
    peak_tau_default = cfg.get("infer", {}).get("peak_tau", 1.0)
    peak_mode_default = cfg.get("loss", {}).get("peak_mode", "logsumexp")
    peak_mode = str(args.peak_mode) if args.peak_mode is not None else str(peak_mode_default or "logsumexp")
    if peak_mode not in {"logsumexp", "energy"}:
        raise ValueError(f"Invalid peak_mode: {peak_mode}")
    if presence_mode != "energy" and peak_mode == "energy":
        logger.warning("peak_mode=energy ignored because presence_mode is not energy; falling back to logsumexp.")
        peak_mode = "logsumexp"
    peak_logit_alpha = float(cfg.get("loss", {}).get("peak_logit_alpha", 0.5) or 0.5)
    peak_logit_center = str(cfg.get("loss", {}).get("peak_logit_center", "none") or "none")
    presence_topk_default = cfg.get("train", {}).get("presence_topk", 0)
    peak_pool = str(args.peak_pool) if args.peak_pool is not None else str(peak_pool_default or "max")
    peak_tau = float(args.peak_tau) if args.peak_tau is not None else float(peak_tau_default or 1.0)
    presence_topk_cfg = int(args.presence_topk) if args.presence_topk is not None else int(presence_topk_default or 0)
    presence_topk = presence_topk_cfg if presence_topk_cfg > 0 else None
    peak_threshold_default = cfg.get("infer", {}).get("peak_threshold", None)
    if presence_from_peak and args.peak_threshold is not None:
        threshold_for_metrics = args.peak_threshold
    elif presence_from_peak and args.peak_threshold is None and peak_threshold_default is not None:
        threshold_for_metrics = float(peak_threshold_default)
    else:
        threshold_for_metrics = args.threshold if args.threshold is not None else presence_threshold_default
    roi_base_radius = args.roi_base_radius or cfg.get("infer", {}).get("roi_base_radius_px", 112)
    roi_sigma_multiplier = args.roi_sigma_multiplier or cfg.get("infer", {}).get("roi_sigma_multiplier", 2.5)

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    manifest_df = load_manifest(args.manifest_csv, logger)
    if manifest_df.empty:
        logger.error("Manifest is empty after filtering, aborting")
        return

    if args.backbone:
        cfg.setdefault("train", {})["backbone"] = args.backbone

    logger.info("Temporal window: T=%d, stride=%d", temporal_T, temporal_stride)

    dataset = EvalDataset(
        manifest_df,
        image_size=image_size,
        logger=logger,
        temporal_T=temporal_T,
        temporal_stride=temporal_stride,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
        collate_fn=collate_batch,
    )

    use_energy = presence_mode == "energy"
    model = build_model(
        cfg,
        args.checkpoint,
        device,
        logger,
        temporal_T=temporal_T,
        heatmap_stride=stride,
        use_energy_fusion=use_energy,
    )
    center_tau_default = cfg.get("infer", {}).get("center_tau", None)
    if center_tau_default is None:
        center_tau_default = cfg.get("loss", {}).get("dsnt_tau", 1.0)
    soft_argmax_tau = float(args.soft_argmax_tau) if args.soft_argmax_tau is not None else float(center_tau_default or 1.0)

    preds = run_inference(
        model,
        loader,
        device,
        stride,
        args.soft_argmax,
        soft_argmax_tau,
        args.amp,
        presence_mode=presence_mode,
        peak_pool=peak_pool,
        peak_tau=peak_tau,
        presence_topk=presence_topk,
        peak_logit_alpha=peak_logit_alpha,
        peak_logit_center=peak_logit_center,
        peak_mode=peak_mode,
        dsnt_tau=float(cfg.get("loss", {}).get("dsnt_tau", 1.0)),
    )
    preds_df = pd.DataFrame(preds).sort_values("manifest_idx").reset_index(drop=True)
    if args.threshold is not None or args.peak_threshold is not None:
        tau = args.peak_threshold if presence_from_peak and args.peak_threshold is not None else args.threshold
        preds_df["presence_pred"] = (preds_df["presence_prob"] >= tau).astype(int)

    meta_map = None
    if args.letterbox_meta:
        if not os.path.exists(args.letterbox_meta):
            raise FileNotFoundError(f"letterbox meta not found: {args.letterbox_meta}")
        meta_map = load_letterbox_meta(args.letterbox_meta)
        logger.info("Loaded letterbox meta for %d frames", len(meta_map))
        preds_df = attach_original_xy(
            preds_df,
            meta_map,
            x_col="x_g",
            y_col="y_g",
            out_x="x_orig",
            out_y="y_orig",
            logger=logger,
        )
        manifest_df = attach_manifest_gt_orig(manifest_df, meta_map, logger)
    if args.export_roi:
        if meta_map is None:
            raise ValueError("--export-roi richiede --letterbox-meta")
        roi_dir = args.roi_dir or os.path.join(out_dir, "roi")
        preds_df = apply_backprojection_and_roi(
            preds_df,
            meta_map,
            roi_dir=roi_dir,
            threshold=threshold_for_metrics,
            roi_base_radius=roi_base_radius,
            roi_sigma_multiplier=roi_sigma_multiplier,
            stride=stride,
        )
    elif meta_map is not None:
        logger.info("ROI export disattivato: meta caricati ma non verranno salvati ritagli.")

    save_preds = args.save_preds or os.path.join(out_dir, "preds.csv")
    os.makedirs(os.path.dirname(save_preds) or out_dir, exist_ok=True)
    preds_to_disk = preds_df.drop(columns=["manifest_idx"])
    if "peak_width_heatmap" in preds_to_disk.columns:
        preds_to_disk = preds_to_disk.drop(columns=["peak_width_heatmap"])
    preds_to_disk.to_csv(save_preds, index=False)
    logger.info("Predictions saved to %s", save_preds)

    metrics_path = args.metrics_out
    curves_dir = args.sweep_curves
    has_presence_gt = "presence" in manifest_df.columns and manifest_df["presence"].notna().any()
    has_center_gt = False
    if {"x_pix_resized", "y_pix_resized"}.issubset(manifest_df.columns):
        has_center_gt = manifest_df["x_pix_resized"].notna().any() and manifest_df["y_pix_resized"].notna().any()
    if not has_center_gt and {"cx", "cy"}.issubset(manifest_df.columns):
        has_center_gt = manifest_df["cx"].notna().any() and manifest_df["cy"].notna().any()
    if metrics_path and not has_presence_gt:
        logger.warning("Metrics requested but manifest lacks presence GT. Skipping metrics.")
        metrics_path = None
    if curves_dir and not has_presence_gt:
        logger.warning("PR/ROC curves requested but manifest lacks presence GT. Skipping curves.")
        curves_dir = None

    need_scores = bool(metrics_path or curves_dir)
    if need_scores:
        joined = preds_df.merge(manifest_df, on="manifest_idx", how="left", suffixes=("", "_gt"))
        if "image_path_y" in joined.columns:
            joined = joined.rename(columns={"image_path_x": "image_path"})
            joined = joined.drop(columns=["image_path_y"])
        # --- Presence GT handling (soft -> binary at 0.5) ---
        # Training can use a probabilistic presence over the temporal span; mirror that here for evaluation by:
        # 1) computing presence_gt_prob over the temporal window
        # 2) binarizing it at 0.5 for PR/ROC/AUPRC.
        def _build_presence_map(df: pd.DataFrame) -> dict[str, float]:
            if "image_path" not in df.columns or "presence" not in df.columns:
                return {}
            out: dict[str, float] = {}
            for r in df.itertuples(index=False):
                try:
                    p = os.path.abspath(getattr(r, "image_path"))
                    out[p] = float(getattr(r, "presence"))
                except Exception:
                    continue
            return out

        def _presence_probability(window_paths: list[str], selector: TemporalWindowSelector, presence_map: dict[str, float], default_presence: float = 0.0) -> float:
            if not window_paths:
                return float(default_presence)
            dir_path = os.path.dirname(window_paths[0])
            selector._ensure_dir(dir_path)
            files = selector._dir_cache.get(dir_path, [])
            idx_map = selector._dir_index.get(dir_path, {})
            indices = [idx_map.get(os.path.basename(p)) for p in window_paths if os.path.basename(p) in idx_map]
            if not indices:
                return float(default_presence)
            start, end = min(indices), max(indices)
            values = []
            for i in range(start, end + 1):
                abs_path = os.path.abspath(files[i])
                values.append(presence_map.get(abs_path, default_presence))
            if not values:
                return float(default_presence)
            return float(np.mean(values))

        presence_map = _build_presence_map(manifest_df)
        selector = TemporalWindowSelector(temporal_T=temporal_T, temporal_stride=temporal_stride)
        presence_gt_prob = []
        for p in joined["image_path"].tolist():
            w = selector.get_window(p)
            presence_gt_prob.append(_presence_probability(w, selector, presence_map, default_presence=0.0))
        joined["presence_gt_prob"] = np.asarray(presence_gt_prob, dtype=float)
        y_true = (joined["presence_gt_prob"].to_numpy() >= 0.5).astype(int)

        y_score_combined = joined["presence_prob"].to_numpy()
        y_score_logit = joined["presence_prob_raw"].to_numpy() if "presence_prob_raw" in joined.columns else None
        if {"x_pix_resized", "y_pix_resized"}.issubset(joined.columns):
            finite_centers = np.isfinite(joined["x_pix_resized"]) & np.isfinite(joined["y_pix_resized"])
        else:
            finite_centers = np.zeros(len(joined), dtype=bool)
        pr_curve_comb = metrics_lib.pr_curve(y_true, y_score_combined)
        roc_curve_comb = metrics_lib.roc_curve(y_true, y_score_combined)
        pr_curve_logit = roc_curve_logit = None
        if y_score_logit is not None:
            pr_curve_logit = metrics_lib.pr_curve(y_true, y_score_logit)
            roc_curve_logit = metrics_lib.roc_curve(y_true, y_score_logit)

        if metrics_path:
            os.makedirs(os.path.dirname(metrics_path) or out_dir, exist_ok=True)
            metrics_payload = {
                "dataset_size": int(len(joined)),
                "n_with_gt_presence": int(np.isfinite(y_true).sum()),
                "n_with_gt_center": int(finite_centers.sum()),
                "threshold_used": float(threshold_for_metrics) if threshold_for_metrics is not None else None,
                "gt_presence_rule": "presence_gt_prob>=0.5 (temporal-span mean over window)",
            }
            metrics_payload["presence_metrics_combined"] = metrics_lib.presence_aggregate(
                y_true,
                y_score_combined,
                threshold_for_metrics,
                pr_curve_comb,
                roc_curve_comb,
            )
            if y_score_logit is not None:
                metrics_payload["presence_metrics_logit"] = metrics_lib.presence_aggregate(
                    y_true,
                    y_score_logit,
                    threshold_for_metrics,
                    pr_curve_logit,
                    roc_curve_logit,
                )
            if has_center_gt:
                joined["x_g"] = joined["x_g"].astype(float)
                joined["y_g"] = joined["y_g"].astype(float)
                joined["x_pix_resized"] = joined["x_pix_resized"].astype(float)
                joined["y_pix_resized"] = joined["y_pix_resized"].astype(float)
                gt_centers_mask = finite_centers
                presence_mask = joined["presence"].fillna(0) == 1
                if args.oracle_localization:
                    eval_mask = presence_mask & gt_centers_mask
                    localization_policy = "oracle"
                else:
                    preds_mask = joined["presence_prob"] >= threshold_for_metrics
                    eval_mask = presence_mask & gt_centers_mask & preds_mask
                    localization_policy = "detection_first"
                mask_array = eval_mask.to_numpy(dtype=bool)
                errors_px = metrics_lib.center_errors_px(
                    joined["x_g"].to_numpy(),
                    joined["y_g"].to_numpy(),
                    joined["x_pix_resized"].to_numpy(),
                    joined["y_pix_resized"].to_numpy(),
                    mask_array,
                )
                metrics_payload["center_metrics_px"] = metrics_lib.center_metrics_summary(
                    errors_px,
                    thresholds=args.center_thresholds_px,
                    unit="px",
                    policy=localization_policy,
                )
                geo_metrics = compute_geo_center_metrics(
                    joined,
                    mask_array,
                    thresholds_km=args.center_thresholds_km,
                    localization_policy=localization_policy,
                    logger=logger,
                )
                if geo_metrics is not None:
                    metrics_payload["center_metrics_km"] = geo_metrics
            with open(metrics_path, "w") as f:
                json.dump(metrics_payload, f, indent=2)
            logger.info("Metrics saved to %s", metrics_path)

        if curves_dir:
            os.makedirs(curves_dir, exist_ok=True)
            pr_path = os.path.join(curves_dir, "pr_curve_combined.csv")
            roc_path = os.path.join(curves_dir, "roc_curve_combined.csv")
            pd.DataFrame(pr_curve_comb).to_csv(pr_path, index=False)
            pd.DataFrame(roc_curve_comb).to_csv(roc_path, index=False)
            if pr_curve_logit is not None and roc_curve_logit is not None:
                pd.DataFrame(pr_curve_logit).to_csv(os.path.join(curves_dir, "pr_curve_logit.csv"), index=False)
                pd.DataFrame(roc_curve_logit).to_csv(os.path.join(curves_dir, "roc_curve_logit.csv"), index=False)
            logger.info("Saved sweep curves to %s", curves_dir)


if __name__ == "__main__":
    main()
