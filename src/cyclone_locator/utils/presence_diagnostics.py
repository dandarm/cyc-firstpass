import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cyclone_locator import metrics as metrics_lib
from cyclone_locator.datasets.med_fullbasin import MedFullBasinDataset
from cyclone_locator.infer import build_model, compute_peak_logit
from cyclone_locator.models.energy_fusion import compute_energy_features


def _smooth_targets(target: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0.0:
        return target
    return target * (1.0 - eps) + (1.0 - target) * eps


def _resolve_presence_mode(cfg_run: dict, presence_mode: Optional[str]) -> str:
    if presence_mode:
        mode = str(presence_mode)
    else:
        mode = str(cfg_run.get("infer", {}).get("presence_mode") or cfg_run.get("train", {}).get("presence_mode") or "")
    mode = mode.lower().strip()
    if not mode:
        mode = "peak" if bool(cfg_run.get("train", {}).get("presence_from_peak", False)) else "head"
    if mode not in {"head", "peak", "both", "energy"}:
        raise ValueError(f"Invalid presence_mode: {mode}")
    return mode


def _resolve_peak_mode(cfg_run: dict, presence_mode: str, peak_mode: Optional[str]) -> str:
    mode = str(peak_mode) if peak_mode is not None else str(cfg_run.get("loss", {}).get("peak_mode", "logsumexp"))
    mode = mode.lower().strip()
    if mode not in {"logsumexp", "energy"}:
        raise ValueError(f"Invalid peak_mode: {mode}")
    if presence_mode != "energy" and mode == "energy":
        mode = "logsumexp"
    return mode


def _presence_mode_for_checkpoint(cfg_run: dict, checkpoint_path: Path, presence_mode: str) -> bool:
    if presence_mode == "energy":
        return True
    state = torch.load(checkpoint_path, map_location="cpu")
    weights = state.get("model", state)
    return any(str(k).startswith("energy_fusion.") for k in weights.keys())


def _build_presence_mapping(df: pd.DataFrame) -> Dict[str, Tuple[int, str]]:
    df = df.copy()
    df["manifest_idx"] = np.arange(len(df))
    df["image_path_abs"] = df["image_path"].apply(lambda p: str(Path(p).resolve()))
    event_ids = df["event_id"].astype(str) if "event_id" in df.columns else pd.Series([""] * len(df))
    return {p: (int(idx), str(evt)) for p, idx, evt in zip(df["image_path_abs"], df["manifest_idx"], event_ids)}


def _tail_summary(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return {
            "top_1p": float("nan"),
            "top_5p": float("nan"),
            "top_10p": float("nan"),
            "rest_90p": float("nan"),
        }
    values_sorted = np.sort(values)[::-1]
    def _mean_top(frac: float) -> float:
        k = max(1, int(math.ceil(frac * n)))
        return float(values_sorted[:k].mean())
    k10 = max(1, int(math.ceil(0.10 * n)))
    rest = values_sorted[k10:]
    rest_mean = float(rest.mean()) if len(rest) else float("nan")
    return {
        "top_1p": _mean_top(0.01),
        "top_5p": _mean_top(0.05),
        "top_10p": _mean_top(0.10),
        "rest_90p": rest_mean,
    }


def export_presence_diagnostics(
    *,
    cfg_run: dict,
    checkpoint_path: str | Path,
    manifest_csv: str | Path,
    out_dir: str | Path,
    split_name: str,
    device: Optional[torch.device] = None,
    presence_mode: Optional[str] = None,
    peak_mode: Optional[str] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    dataset_manifest_stride: Optional[int] = None,
    max_batches: Optional[int] = None,
    topn: int = 50,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_path)
    manifest_csv = Path(manifest_csv)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    presence_mode = _resolve_presence_mode(cfg_run, presence_mode)
    peak_mode = _resolve_peak_mode(cfg_run, presence_mode, peak_mode)
    use_energy_fusion = _presence_mode_for_checkpoint(cfg_run, checkpoint_path, presence_mode)

    train_cfg = cfg_run.get("train", {})
    loss_cfg = cfg_run.get("loss", {})
    data_cfg = cfg_run.get("data", {})
    infer_cfg = cfg_run.get("infer", {})

    temporal_T = int(train_cfg.get("temporal_T", 1) or 1)
    temporal_stride = int(train_cfg.get("temporal_stride", 1) or 1)
    image_size = int(train_cfg.get("image_size", 224))
    heatmap_stride = int(train_cfg.get("heatmap_stride", 4))
    manifest_stride = int(dataset_manifest_stride or data_cfg.get("manifest_stride", 1) or 1)
    presence_topk_cfg = int(train_cfg.get("presence_topk", 0) or 0)
    presence_topk = presence_topk_cfg if presence_topk_cfg > 0 else None
    peak_pool = str(loss_cfg.get("peak_pool", "max") or "max")
    peak_tau = float(loss_cfg.get("peak_tau", 1.0) or 1.0)
    peak_logit_center = str(loss_cfg.get("peak_logit_center", "none") or "none")
    peak_logit_alpha = float(loss_cfg.get("peak_logit_alpha", 0.5) or 0.5)
    dsnt_tau = float(loss_cfg.get("dsnt_tau", 1.0) or 1.0)
    presence_smoothing = float(loss_cfg.get("presence_label_smoothing", 0.0) or 0.0)

    if batch_size is None:
        batch_size = int(infer_cfg.get("batch_size", 32))
    if num_workers is None:
        num_workers = int(train_cfg.get("num_workers", 0))

    df_full = pd.read_csv(manifest_csv)
    mapping = _build_presence_mapping(df_full)

    dataset = MedFullBasinDataset(
        csv_path=str(manifest_csv),
        image_size=image_size,
        heatmap_stride=heatmap_stride,
        heatmap_sigma_px=float(loss_cfg.get("heatmap_sigma_px", 8.0)),
        use_aug=False,
        use_pre_letterboxed=bool(data_cfg.get("use_pre_letterboxed", True)),
        letterbox_meta_csv=str(Path(data_cfg.get("letterbox_meta_csv", "")))
        if data_cfg.get("letterbox_meta_csv")
        else None,
        letterbox_size_assert=data_cfg.get("letterbox_size_assert"),
        temporal_T=temporal_T,
        temporal_stride=temporal_stride,
        manifest_stride=manifest_stride,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        drop_last=False,
    )

    import logging
    logger = logging.getLogger("presence_diagnostics")
    logger.setLevel(logging.INFO)

    model = build_model(
        cfg_run,
        str(checkpoint_path),
        device,
        logger,
        temporal_T=temporal_T,
        heatmap_stride=heatmap_stride,
        use_energy_fusion=use_energy_fusion,
    )
    energy_fusion = getattr(model, "energy_fusion", None)

    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            input_key = "video" if getattr(model, "input_is_video", False) else "image"
            images = batch[input_key].to(device)
            pres = batch["presence"].to(device).float()
            pres_smooth = _smooth_targets(pres, presence_smoothing)
            heatmaps_pred, logits = model(images)
            heatmaps_pred = torch.nan_to_num(heatmaps_pred, nan=0.0, posinf=50.0, neginf=-50.0)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

            E, C = compute_energy_features(heatmaps_pred, dsnt_tau=float(dsnt_tau), topk=presence_topk)
            z_head = logits

            peak_logit = None
            if presence_mode == "energy":
                if energy_fusion is None:
                    raise ValueError("energy_fusion module not found on model (presence_mode=energy)")
                z_tot = energy_fusion(E, C, z_head)
            elif presence_mode == "both":
                peak_logit = compute_peak_logit(
                    heatmaps_pred,
                    pool=peak_pool,
                    tau=float(peak_tau),
                    topk=presence_topk,
                    center_mode=peak_logit_center,
                )
                z_tot = z_head + float(peak_logit_alpha) * peak_logit
            elif presence_mode == "peak":
                peak_logit = compute_peak_logit(
                    heatmaps_pred,
                    pool=peak_pool,
                    tau=float(peak_tau),
                    topk=presence_topk,
                    center_mode=peak_logit_center,
                )
                z_tot = peak_logit
            else:
                z_tot = z_head

            bce_clip = F.binary_cross_entropy_with_logits(z_tot, pres_smooth, reduction="none").squeeze(1)
            p_tot = torch.sigmoid(z_tot).squeeze(1)

            image_paths = batch.get("image_path", [""] * len(p_tot))
            image_paths_abs = batch.get("image_path_abs", image_paths)

            for i in range(len(p_tot)):
                abs_path = str(Path(image_paths_abs[i]).resolve())
                manifest_idx, event_id = mapping.get(abs_path, (-1, ""))
                rows.append({
                    "clip_id": int(manifest_idx),
                    "image_path": str(image_paths[i]),
                    "event_id": str(event_id),
                    "presence_label": float(pres[i].item()),
                    "z_tot": float(z_tot[i].item()),
                    "p_tot": float(p_tot[i].item()),
                    "bce_clip": float(bce_clip[i].item()),
                    "z_head": float(z_head[i].item()),
                    "energy_E": float(E[i].item()) if E is not None else float("nan"),
                    "energy_C": float(C[i].item()) if C is not None else float("nan"),
                })
            if max_batches is not None and (bi + 1) >= max_batches:
                break

    df = pd.DataFrame(rows)
    per_clip_path = out_dir / f"{split_name}_presence_per_clip.csv"
    df.to_csv(per_clip_path, index=False)

    report = {
        "split": split_name,
        "total_clips": int(len(df)),
        "presence_mode": presence_mode,
        "peak_mode": peak_mode,
    }
    report["all"] = _tail_summary(df["bce_clip"].to_numpy())
    for label, name in [(1, "pos"), (0, "neg")]:
        subset = df[df["presence_label"] >= 0.5] if label == 1 else df[df["presence_label"] < 0.5]
        report[name] = _tail_summary(subset["bce_clip"].to_numpy())

    topn_df = df.sort_values("bce_clip", ascending=False).head(int(topn))
    topn_path = out_dir / f"{split_name}_presence_tail_top{int(topn)}.csv"
    topn_df.to_csv(topn_path, index=False)

    report_path = out_dir / f"{split_name}_presence_tail_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    event_rows = []
    for event_id, group in df.groupby("event_id"):
        bce_vals = group["bce_clip"].to_numpy()
        y_true = (group["presence_label"].to_numpy() >= 0.5).astype(int)
        y_score = group["p_tot"].to_numpy()
        pr_curve = metrics_lib.pr_curve(y_true, y_score)
        roc_curve = metrics_lib.roc_curve(y_true, y_score)
        event_rows.append({
            "event_id": event_id,
            "count": int(len(group)),
            "count_pos": int((y_true == 1).sum()),
            "count_neg": int((y_true == 0).sum()),
            "mean_bce": float(np.nanmean(bce_vals)) if len(bce_vals) else float("nan"),
            "median_bce": float(np.nanmedian(bce_vals)) if len(bce_vals) else float("nan"),
            "p90_bce": float(np.nanpercentile(bce_vals, 90)) if len(bce_vals) else float("nan"),
            "auprc": float(pr_curve[0]["pr_auc"]) if pr_curve else float("nan"),
            "roc_auc": float(roc_curve[0]["roc_auc"]) if roc_curve else float("nan"),
        })

    per_event_path = out_dir / f"{split_name}_presence_per_event.csv"
    pd.DataFrame(event_rows).to_csv(per_event_path, index=False)

    return {
        "per_clip_csv": per_clip_path,
        "tail_report": report_path,
        "tail_topn_csv": topn_path,
        "per_event_csv": per_event_path,
    }
