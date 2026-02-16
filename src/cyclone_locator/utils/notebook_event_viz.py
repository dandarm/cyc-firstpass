from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess

from cyclone_locator.datasets.med_fullbasin import MedFullBasinDataset
from cyclone_locator.infer import decode_heatmap, spatial_peak_pool
from cyclone_locator.utils.event_sampling import compute_presence_gt_prob, load_manifest_csv


@dataclass(frozen=True)
class SampleViz:
    frames: np.ndarray  # (T,H,W,C) in [0,1]
    hm_pred: np.ndarray  # (Hh,Wh) in [0,1] (sigmoid)
    hm_tgt: np.ndarray  # (Hh,Wh) in [0,1]
    tx: int
    ty: int
    px: int
    py: int


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        frame = frame[..., None]
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return frame


def make_video_mosaic(frames: np.ndarray, ncols: int = 4, pad: int = 2, pad_value: float = 0.0) -> np.ndarray:
    """
    frames: (T,H,W,C) in [0,1]
    Returns a mosaic image for quick inspection.
    """
    if frames.ndim != 4:
        raise ValueError(f"Expected frames (T,H,W,C), got {frames.shape}")
    t, h, w, c = frames.shape
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(t / ncols))
    out_h = nrows * h + (nrows - 1) * pad
    out_w = ncols * w + (ncols - 1) * pad
    mosaic = np.full((out_h, out_w, c), pad_value, dtype=np.float32)
    for i in range(t):
        r = i // ncols
        cc = i % ncols
        y0 = r * (h + pad)
        x0 = cc * (w + pad)
        mosaic[y0 : y0 + h, x0 : x0 + w] = frames[i]
    return mosaic


def compute_sample_viz(
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    dataset_idx: int,
) -> SampleViz:
    sample = dataset[int(dataset_idx)]
    video = sample["video"].unsqueeze(0).to(device)
    image = sample["image"].unsqueeze(0).to(device)
    inp = video if getattr(model, "input_is_video", False) else image
    with torch.no_grad():
        hm_pred, _ = model(inp)
    hm_pred = hm_pred.squeeze(0).squeeze(0)
    hm_pred_vis = torch.sigmoid(hm_pred).float().cpu().numpy()
    hm_tgt = sample["heatmap"].squeeze(0).squeeze(0).float().cpu().numpy()
    py, px = np.unravel_index(int(np.argmax(hm_pred_vis)), hm_pred_vis.shape)
    ty, tx = np.unravel_index(int(np.argmax(hm_tgt)), hm_tgt.shape)

    frames = sample["video"].permute(1, 2, 3, 0).float().cpu().numpy()  # (T,H,W,C)
    frames = np.clip(frames, 0.0, 1.0)
    frames = np.stack([_to_rgb(frames[i]) for i in range(frames.shape[0])], axis=0)
    return SampleViz(frames=frames, hm_pred=hm_pred_vis, hm_tgt=hm_tgt, tx=tx, ty=ty, px=px, py=py)


def plot_event_samples_grid(
    event_df,
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    *,
    heatmap_stride: float,
    save_dir: Optional[str | Path] = None,
    show: bool = True,
    mosaic_cols: int = 4,
):
    """
    event_df: DataFrame subset for a single event_id, expected columns: phase, dataset_idx, presence_gt_prob, timestamp.
    Plots 5 rows (phases) with 3 panels: video mosaic | GT heatmap | predicted heatmap.
    """
    if event_df is None or len(event_df) == 0:
        return None

    phases = ["before_start", "transition", "middle", "final", "after_end"]
    # Keep only these phases, in order
    rows = []
    for ph in phases:
        sub = event_df[event_df["phase"] == ph]
        if len(sub) == 0:
            continue
        rows.append(sub.iloc[0])
    if not rows:
        return None

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.2 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for r_i, r in enumerate(rows):
        viz = compute_sample_viz(dataset, model, device, int(r["dataset_idx"]))
        mosaic = make_video_mosaic(viz.frames, ncols=mosaic_cols)

        ax_vid, ax_gt, ax_pr = axes[r_i, 0], axes[r_i, 1], axes[r_i, 2]
        ax_vid.imshow(mosaic)
        ax_vid.set_title(f"{r['phase']} | idx={int(r['dataset_idx'])} | p={float(r['presence_gt_prob']):.3f} | {r.get('timestamp','')}")
        ax_vid.axis("off")

        im_gt = ax_gt.imshow(viz.hm_tgt, cmap="magma", vmin=0, vmax=1, interpolation="bilinear")
        ax_gt.scatter([viz.tx], [viz.ty], c="cyan", s=40, marker="x", label="tgt argmax")
        ax_gt.set_title("GT heatmap")
        ax_gt.axis("off")
        ax_gt.legend(loc="lower left")
        fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

        im_pr = ax_pr.imshow(viz.hm_pred, cmap="magma", vmin=0, vmax=1, interpolation="bilinear")
        ax_pr.scatter([viz.px], [viz.py], c="lime", s=40, marker="x", label="pred argmax")
        ax_pr.scatter([viz.tx], [viz.ty], c="cyan", s=40, marker="x", label="tgt argmax")
        ax_pr.set_title("Pred heatmap (sigmoid)")
        ax_pr.axis("off")
        ax_pr.legend(loc="lower left")
        fig.colorbar(im_pr, ax=ax_pr, fraction=0.046, pad=0.04)

    fig.suptitle(f"Event {str(rows[0].get('event_id',''))} | heatmap_stride={heatmap_stride}", y=0.995)
    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        event_id = str(rows[0].get("event_id", "event"))
        out_path = save_dir / f"event_samples_{event_id}.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _as_rgb_uint8(img01: np.ndarray) -> np.ndarray:
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255.0).round().astype(np.uint8)


def _render_three_panel_frame(
    frame_rgb01: np.ndarray,
    hm_gt01: np.ndarray,
    hm_pr01: np.ndarray,
    *,
    title: str,
    gt_xy_hm: Tuple[float, float],
    pr_xy_hm: Tuple[float, float],
    hm_pred_vmax: float = 1.0,
    bottom_lines: Optional[Sequence[str]] = None,
    status_text: Optional[str] = None,
    status_color: Tuple[int, int, int] = (0, 255, 0),
    series_vals: Optional[Sequence[float]] = None,
    series_index: Optional[int] = None,
    series_label: str = "",
    series_color: Tuple[int, int, int] = (255, 255, 0),
    series_marks: Optional[Sequence[Tuple[int, float]]] = None,
    dpi: int = 140,
) -> np.ndarray:
    fig = plt.figure(figsize=(12, 4), dpi=dpi)
    canvas = FigureCanvas(fig)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.08)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    ax0.imshow(frame_rgb01)
    ax0.set_title(title, fontsize=10)
    ax0.axis("off")

    im1 = ax1.imshow(hm_gt01, cmap="magma", vmin=0.0, vmax=1.0, interpolation="bilinear")
    ax1.scatter([gt_xy_hm[0]], [gt_xy_hm[1]], c="cyan", s=30, marker="x", label="GT")
    ax1.set_title("GT heatmap", fontsize=10)
    ax1.axis("off")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    hm_pred_vmax = float(np.clip(hm_pred_vmax, 1e-6, 1.0))
    im2 = ax2.imshow(hm_pr01, cmap="magma", vmin=0.0, vmax=hm_pred_vmax, interpolation="bilinear")
    ax2.scatter([pr_xy_hm[0]], [pr_xy_hm[1]], c="lime", s=30, marker="x", label="Pred")
    ax2.scatter([gt_xy_hm[0]], [gt_xy_hm[1]], c="cyan", s=30, marker="x", label="GT")
    ax2.set_title("Pred heatmap (sigmoid)", fontsize=10)
    ax2.axis("off")
    ax2.legend(loc="lower left")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    rgb = buf[..., :3].copy()
    plt.close(fig)
    return rgb


def _render_three_panel_frame_cv2(
    frame_rgb01: np.ndarray,
    hm_gt01: np.ndarray,
    hm_pr01: np.ndarray,
    *,
    title: str,
    gt_xy_hm: Tuple[float, float],
    pr_xy_hm: Tuple[float, float],
    center_xy_frame: Optional[Tuple[float, float]] = None,
    center_xy_hm: Optional[Tuple[float, float]] = None,
    hm_pred_vmax: float = 1.0,
    bottom_lines: Optional[Sequence[str]] = None,
    status_text: Optional[str] = None,
    status_color: Tuple[int, int, int] = (0, 255, 0),
    series_vals: Optional[Sequence[float]] = None,
    series_index: Optional[int] = None,
    series_label: str = "",
    series_color: Tuple[int, int, int] = (255, 255, 0),
    series_marks: Optional[Sequence[Tuple[int, float]]] = None,
    gt_alpha: float = 1.0,
    gt_mask_rel: float = 0.0,
    pad: int = 6,
) -> np.ndarray:
    """
    Faster renderer using OpenCV if available.
    Returns uint8 RGB image.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        # Fallback to matplotlib renderer.
        return _render_three_panel_frame(
            frame_rgb01,
            hm_gt01,
            hm_pr01,
            title=title,
            gt_xy_hm=gt_xy_hm,
            pr_xy_hm=pr_xy_hm,
            hm_pred_vmax=hm_pred_vmax,
            bottom_lines=bottom_lines,
            status_text=status_text,
            status_color=status_color,
            series_vals=series_vals,
            series_index=series_index,
            series_label=series_label,
            series_color=series_color,
            series_marks=series_marks,
        )

    frame_u8 = _as_rgb_uint8(frame_rgb01)
    frame_u8 = np.ascontiguousarray(frame_u8)
    h, w = frame_u8.shape[:2]

    def _heatmap_to_rgb(hm01: np.ndarray, vmax: float) -> np.ndarray:
        vmax = float(np.clip(vmax, 1e-6, 1.0))
        x = np.clip(hm01 / vmax, 0.0, 1.0)
        x_u8 = (x * 255.0).round().astype(np.uint8)
        cmap = getattr(cv2, "COLORMAP_MAGMA", None)
        if cmap is None:
            cmap = getattr(cv2, "COLORMAP_INFERNO", 11)
        rgb = cv2.applyColorMap(x_u8, cmap)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(rgb)

    # Pred heatmap as red-only on black
    hm_pr_vmax = float(np.clip(hm_pred_vmax, 1e-6, 1.0))
    hm_pr_resized = cv2.resize(hm_pr01.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    hm_pr_rel = np.clip(hm_pr_resized / hm_pr_vmax, 0.0, 1.0)
    hm_pr_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    hm_pr_rgb[..., 0] = (hm_pr_rel * 255.0).round().astype(np.uint8)

    # Overlay GT heatmap on the frame (zeros/near-zeros transparent).
    gt_alpha = float(np.clip(gt_alpha, 0.0, 1.0))
    gt_mask_rel = float(np.clip(gt_mask_rel, 0.0, 1.0))
    hm_gt_resized = cv2.resize(hm_gt01.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    gt_max = float(np.max(hm_gt_resized)) if hm_gt_resized.size else 0.0
    if gt_max > 0.0 and gt_alpha > 0.0:
        # Match notebook: alpha proportional to intensity, zeros fully transparent.
        thr = max(0.0, gt_mask_rel * gt_max)
        alpha_map = (hm_gt_resized / max(gt_max, 1e-6)).astype(np.float32) * float(gt_alpha)
        alpha_map[hm_gt_resized <= thr] = 0.0
        alpha3 = np.repeat(alpha_map[..., None], 3, axis=-1)
        base = frame_u8.astype(np.float32)
        green = np.zeros_like(base)
        green[..., 1] = 255.0  # RGB green
        frame_u8 = np.clip(base * (1.0 - alpha3) + green * alpha3, 0.0, 255.0).astype(np.uint8)
        frame_u8 = np.ascontiguousarray(frame_u8)

    # Draw marker in predicted heatmap panel (in resized space).
    # NOTE: `hm_pr_rgb` is an RGB array, so colors are specified in RGB order.
    def _draw_cross(img: np.ndarray, x: float, y: float, color_rgb: Tuple[int, int, int]) -> None:
        hs, ws = hm_gt01.shape[:2]
        sx = w / float(ws)
        sy = h / float(hs)
        cx = int(round(x * sx))
        cy = int(round(y * sy))
        size = 6
        cv2.drawMarker(img, (cx, cy), color=color_rgb, markerType=cv2.MARKER_TILTED_CROSS, markerSize=size, thickness=2)

    # Pred only: red cross
    _draw_cross(hm_pr_rgb, pr_xy_hm[0], pr_xy_hm[1], (255, 0, 0))

    # Markers on overlayed frame
    hs, ws = hm_gt01.shape[:2]
    sx = w / float(ws)
    sy = h / float(hs)
    gt_x_px = int(round(float(gt_xy_hm[0]) * sx))
    gt_y_px = int(round(float(gt_xy_hm[1]) * sy))
    pr_x_px = int(round(float(pr_xy_hm[0]) * sx))
    pr_y_px = int(round(float(pr_xy_hm[1]) * sy))
    # NOTE: `frame_u8` is an RGB array, so colors are specified in RGB order.
    # Pred only: red cross on frame.
    cv2.drawMarker(frame_u8, (pr_x_px, pr_y_px), color=(255, 0, 0), markerType=cv2.MARKER_TILTED_CROSS, markerSize=14, thickness=2)

    # Compose side-by-side with padding
    pad_col = np.zeros((h, pad, 3), dtype=np.uint8)
    out = np.concatenate([frame_u8, pad_col, hm_pr_rgb], axis=1)
    # Title strip
    strip_h = 78
    strip = np.zeros((strip_h, out.shape[1], 3), dtype=np.uint8)
    # Wrap title on 2 lines to avoid truncation
    t1 = title[:110]
    t2 = title[110:220] if len(title) > 110 else ""
    t3 = title[220:330] if len(title) > 220 else ""
    cv2.putText(strip, t1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if t2:
        cv2.putText(strip, t2, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if t3:
        cv2.putText(strip, t3, (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    bottom_lines = list(bottom_lines or [])
    has_series = series_vals is not None and len(series_vals) > 1
    bottom_h = 0
    if bottom_lines or has_series:
        bottom_h = 140
        bottom = np.zeros((bottom_h, out.shape[1], 3), dtype=np.uint8)
        y = 22
        for line in bottom_lines[:3]:
            cv2.putText(bottom, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 20
        if status_text:
            cv2.putText(bottom, status_text, (bottom.shape[1] - 180, bottom_h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

        if has_series:
            vals = np.asarray(series_vals, dtype=np.float32)
            finite = np.isfinite(vals)
            if finite.any():
                vmin = 0.0
                vmax = 12.0
                if vmax <= vmin:
                    vmax = vmin + 1.0
                plot_top = 60
                plot_bottom = bottom_h - 12
                plot_left = 10
                plot_right = bottom.shape[1] - 10
                xs = np.linspace(plot_left, plot_right, len(vals))
                ys = plot_bottom - (vals - vmin) / (vmax - vmin) * float(plot_bottom - plot_top)
                pts = np.stack([xs, ys], axis=1).astype(np.int32)
                cv2.polylines(bottom, [pts], False, series_color, 1, cv2.LINE_AA)
                if series_label:
                    cv2.putText(bottom, series_label, (plot_left, plot_top - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                if series_marks:
                    for idx, val in series_marks:
                        if 0 <= idx < len(vals) and np.isfinite(val):
                            x = int(round(xs[idx]))
                            yv = plot_bottom - (val - vmin) / (vmax - vmin) * float(plot_bottom - plot_top)
                            yv = int(round(np.clip(yv, plot_top, plot_bottom)))
                            cv2.line(bottom, (x, plot_top), (x, plot_bottom), (120, 120, 120), 1)
                            dash = 6
                            gap = 4
                            xx = plot_left
                            while xx < plot_right:
                                cv2.line(bottom, (xx, yv), (min(xx + dash, plot_right), yv), (120, 120, 120), 1)
                                xx += dash + gap
                if series_index is not None:
                    idx = int(series_index)
                    if 0 <= idx < len(vals):
                        x = int(round(xs[idx]))
                        y = int(round(ys[idx]))
                        cv2.line(bottom, (x, plot_top), (x, plot_bottom), (255, 0, 0), 1)
                        cv2.circle(bottom, (x, y), 3, (255, 0, 0), -1)
        out = np.concatenate([strip, out, bottom], axis=0)
    else:
        out = np.concatenate([strip, out], axis=0)
    return out


def export_event_videos_for_manifest(
    *,
    cfg_run: dict,
    checkpoint_path: str | Path,
    manifest_csv: str | Path,
    out_dir: str | Path,
    device: torch.device,
    event_ids: Optional[Sequence[str]] = None,
    fps: int = 6,
    max_frames_per_event: Optional[int] = None,
    dataset_manifest_stride: int = 1,
    presence_threshold: Optional[float] = None,
    batch_size: Optional[int] = None,
    num_workers: int = 4,
    render_workers: int = 8,
) -> Dict[str, Path]:
    """
    Produce a per-event MP4 (or PNG frame folder fallback) for all events in a manifest.
    Each video frame corresponds to one center frame (in chronological order) and shows:
      frame + GT heatmap overlay | predicted heatmap.

    Notes:
    - Uses the same dataset and dataloader logic as training (MedFullBasinDataset), but you can
      set dataset_manifest_stride=1 to inspect all frames regardless of the training stride.
    - If MP4 writer is unavailable, saves PNG frames and prints an ffmpeg command.
    """
    manifest_csv = Path(manifest_csv)
    checkpoint_path = Path(checkpoint_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Assume manifests live under <repo>/manifests/*.csv; resolve other config-relative paths from <repo>.
    repo_root = manifest_csv.parent.parent

    train_cfg = cfg_run.get("train", {})
    loss_cfg = cfg_run.get("loss", {})
    data_cfg = cfg_run.get("data", {})
    infer_cfg = cfg_run.get("infer", {})

    temporal_T = int(train_cfg.get("temporal_T", 1) or 1)
    temporal_stride = int(train_cfg.get("temporal_stride", 1) or 1)
    image_size = int(train_cfg.get("image_size", 224))
    heatmap_stride_cfg = int(train_cfg.get("heatmap_stride", 4))

    # Build dataset (same as training), but allow overriding manifest_stride for inspection.
    ds = MedFullBasinDataset(
        csv_path=str(manifest_csv),
        image_size=image_size,
        heatmap_stride=heatmap_stride_cfg,
        heatmap_sigma_px=float(loss_cfg.get("heatmap_sigma_px", 8.0)),
        use_aug=False,
        use_pre_letterboxed=bool(data_cfg.get("use_pre_letterboxed", True)),
        letterbox_meta_csv=(
            str((repo_root / str(data_cfg.get("letterbox_meta_csv"))))
            if data_cfg.get("letterbox_meta_csv")
            else None
        ),
        letterbox_size_assert=data_cfg.get("letterbox_size_assert"),
        temporal_T=temporal_T,
        temporal_stride=temporal_stride,
        manifest_stride=int(dataset_manifest_stride or 1),
    )
    ds.max_missing_retries = 1

    # Load full manifest to build a timestamp map for titles.
    df_full = load_manifest_csv(manifest_csv)
    ts_map = {str(p): t for p, t in zip(df_full["image_path"], df_full["timestamp"])}

    # Compute dataset-visible presence_gt_prob with the same stride as the dataset.
    df_ds = compute_presence_gt_prob(
        df_full,
        temporal_T=temporal_T,
        temporal_stride=temporal_stride,
        manifest_stride=int(dataset_manifest_stride or 1),
    )
    if "event_id" not in df_ds.columns:
        raise ValueError(f"{manifest_csv} missing event_id column; cannot build event videos.")
    df_ds["event_id"] = df_ds["event_id"].astype(str)
    df_ds["timestamp"] = df_ds["timestamp"].astype("datetime64[ns]")

    if event_ids is None:
        event_ids = sorted(df_ds["event_id"].dropna().unique().tolist())
    else:
        event_ids = [str(e) for e in event_ids]

    # Build model using infer helpers for consistency.
    import logging
    from cyclone_locator.infer import build_model

    logger = logging.getLogger("notebook_event_viz")
    logger.setLevel(logging.INFO)
    presence_mode_cfg = infer_cfg.get("presence_mode") or train_cfg.get("presence_mode", "")
    if not presence_mode_cfg:
        presence_mode_cfg = "peak" if bool(train_cfg.get("presence_from_peak", False)) else "head"
    presence_mode_cfg = str(presence_mode_cfg).lower().strip()
    use_energy_fusion = presence_mode_cfg == "energy"

    if not use_energy_fusion:
        state = torch.load(checkpoint_path, map_location="cpu")
        weights = state.get("model", state)
        has_energy_keys = any(str(k).startswith("energy_fusion.") for k in weights.keys())
        if has_energy_keys:
            logger.warning("Checkpoint contains energy_fusion params; enabling energy_fusion for visualization.")
            use_energy_fusion = True

    model = build_model(
        cfg_run,
        str(checkpoint_path),
        device,
        logger,
        temporal_T=temporal_T,
        heatmap_stride=heatmap_stride_cfg,
        use_energy_fusion=use_energy_fusion,
    )

    peak_pool = str(loss_cfg.get("peak_pool", "max"))
    peak_tau = float(loss_cfg.get("peak_tau", 1.0))
    center_tau = float(infer_cfg.get("center_tau", 1.0))
    soft_argmax = True  # align with infer.sh default
    presence_from_peak = bool(infer_cfg.get("presence_from_peak", train_cfg.get("presence_from_peak", False)))
    presence_topk_cfg = int(train_cfg.get("presence_topk", 0) or 0)
    presence_topk = presence_topk_cfg if presence_topk_cfg > 0 else None
    if presence_threshold is None:
        presence_threshold = float(infer_cfg.get("peak_threshold") or infer_cfg.get("presence_threshold") or 0.5)
    # Visual helper: fixed max so colors are comparable across runs.
    # This does NOT affect any model decision.
    hm_pred_vmax = 1.0
    if batch_size is None:
        batch_size = int(infer_cfg.get("batch_size") or 16)
    batch_size = max(1, int(batch_size))
    num_workers = max(0, int(num_workers))
    render_workers = max(0, int(render_workers))

    def _find_ffmpeg_exe() -> Optional[str]:
        for k in ("IMAGEIO_FFMPEG_EXE", "FFMPEG"):
            v = os.environ.get(k)
            if v and os.path.exists(v) and os.access(v, os.X_OK):
                return v
        # fallback: rely on PATH
        return "ffmpeg"

    def _encode_with_ffmpeg(frames_dir: Path, out_mp4: Path) -> bool:
        ffmpeg_exe = _find_ffmpeg_exe()
        if ffmpeg_exe is None:
            return False
        # Basic sanity check when using PATH fallback
        if ffmpeg_exe != "ffmpeg" and not (os.path.exists(ffmpeg_exe) and os.access(ffmpeg_exe, os.X_OK)):
            return False

        # Build concat demuxer list with absolute paths + explicit per-frame duration.
        # Without durations, ffmpeg may treat the concat stream as having near-zero timestamps
        # and produce an MP4 with only a couple of frames.
        frames_txt = frames_dir / "frames.txt"
        frames = sorted(frames_dir.glob("frame_*.png"))
        if not frames:
            print(f"[WARN] ffmpeg encode skipped: no frames found in {frames_dir}")
            return False
        with open(frames_txt, "w", encoding="utf-8") as f:
            dt = 1.0 / float(max(1, int(fps)))
            for p in frames[:-1]:
                f.write(f"file '{str(p.resolve())}'\n")
                f.write(f"duration {dt:.9f}\n")
            # The last file's duration is ignored by the demuxer; repeat the last file once.
            last = frames[-1].resolve()
            f.write(f"file '{str(last)}'\n")
            f.write(f"file '{str(last)}'\n")

        cmd = [
            ffmpeg_exe,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(frames_txt),
            "-vsync",
            "vfr",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            str(out_mp4),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return True
        except Exception as e:
            print(f"[WARN] ffmpeg encode failed for {out_mp4}: {e}")
            return False

    outputs: Dict[str, Path] = {}
    for ev in event_ids:
        ev_df = df_ds[df_ds["event_id"] == str(ev)].copy()
        if ev_df.empty:
            continue
        ev_df = ev_df.sort_values("timestamp")
        idxs = ev_df["dataset_idx"].astype(int).tolist()
        if max_frames_per_event is not None:
            idxs = idxs[: int(max_frames_per_event)]
        if not idxs:
            continue

        out_mp4 = out_dir / f"event_{ev}.mp4"
        frames_dir = out_dir / f"event_{ev}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Pre-load samples into batches using a DataLoader for parallel IO
        from torch.utils.data import DataLoader, Dataset

        class _IdxWrap(Dataset):
            def __init__(self, base_ds, indices):
                self.base_ds = base_ds
                self.indices = list(indices)
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, i):
                ds_idx = int(self.indices[i])
                s = self.base_ds[ds_idx]
                s["dataset_idx"] = ds_idx
                return s

        wrapped = _IdxWrap(ds, idxs)
        loader_series = DataLoader(
            wrapped,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        peak_series = []
        pred_prob_series = []
        gt_prob_series = []
        with torch.no_grad():
            for batch in loader_series:
                input_key = "video" if getattr(model, "input_is_video", False) else "image"
                inp = batch[input_key].to(device, non_blocking=True)
                hm_logits_b, pres_logit_b = model(inp)
                hm_logits_b = hm_logits_b.squeeze(1).detach().float().cpu()
                pres_logit_b = pres_logit_b.squeeze(1).detach().float().cpu()
                peak_logit_pool = spatial_peak_pool(
                    hm_logits_b,
                    pool=peak_pool,
                    tau=float(peak_tau),
                    topk=presence_topk,
                )
                peak_logsumexp = spatial_peak_pool(
                    hm_logits_b,
                    pool="logsumexp",
                    tau=float(peak_tau),
                    topk=presence_topk,
                )
                if presence_from_peak:
                    pred_prob = torch.sigmoid(peak_logit_pool).clamp(0.0, 1.0)
                else:
                    pred_prob = torch.sigmoid(pres_logit_b).clamp(0.0, 1.0)
                peak_series.extend(peak_logsumexp.numpy().tolist())
                pred_prob_series.extend(pred_prob.numpy().tolist())
                gt_prob_series.extend(batch["presence"].squeeze(1).numpy().astype(np.float32).tolist())

        series_marks = []
        gt_pos_idx = [i for i, v in enumerate(gt_prob_series) if float(v) > 0.0]
        if gt_pos_idx:
            start_idx = gt_pos_idx[0]
            end_idx = gt_pos_idx[-1]
            if 0 <= start_idx < len(peak_series):
                series_marks.append((start_idx, float(peak_series[start_idx])))
            if 0 <= end_idx < len(peak_series):
                series_marks.append((end_idx, float(peak_series[end_idx])))

        loader = DataLoader(
            wrapped,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        pool = ThreadPoolExecutor(max_workers=render_workers) if render_workers > 0 else None
        try:
            frame_counter = 0
            for batch in loader:
                ds_idx_batch = batch["dataset_idx"].tolist()
                input_key = "video" if getattr(model, "input_is_video", False) else "image"
                inp = batch[input_key].to(device, non_blocking=True)
                with torch.no_grad():
                    hm_logits_b, _ = model(inp)
                hm_logits_b = hm_logits_b.squeeze(1).detach().float().cpu()  # (B,Hh,Wh)
                hm_pred_prob_b = torch.sigmoid(hm_logits_b).numpy()
                hm_tgt_b = batch["heatmap"].squeeze(1).numpy()  # (B,Hh,Wh)
                video_b = batch["video"]  # (B,C,T,H,W) on CPU

                tasks = []
                for bi, ds_idx in enumerate(ds_idx_batch):
                    v = video_b[bi]  # (C,T,H,W)
                    t = int(v.shape[1])
                    center_i = int(t // 2)
                    frame = v[:, center_i].permute(1, 2, 0).float().cpu().numpy()
                    frame = _to_rgb(np.clip(frame, 0.0, 1.0))

                    hm_pred_prob = hm_pred_prob_b[bi]
                    hm_tgt = hm_tgt_b[bi]
                    ty, tx = np.unravel_index(int(np.argmax(hm_tgt)), hm_tgt.shape)
                    py, px = np.unravel_index(int(np.argmax(hm_pred_prob)), hm_pred_prob.shape)

                    series_idx = frame_counter
                    presence_gt = float(gt_prob_series[series_idx])
                    pred_prob = float(pred_prob_series[series_idx])
                    peak_logsumexp = float(peak_series[series_idx])
                    gt_pos = presence_gt > 0.0
                    pred_pos = pred_prob >= float(presence_threshold)
                    if gt_pos:
                        status_text = "detected" if pred_pos else "missed"
                        status_color = (0, 255, 0) if status_text == "detected" else (255, 0, 0)
                    else:
                        status_text = None
                        status_color = (0, 255, 0)

                    # Center-frame keypoint (if present) in letterbox pixels and heatmap coords
                    abs_path = str(batch["image_path_abs"][bi])
                    center_xy_lb = None
                    try:
                        center_xy_lb = ds._keypoint_lb_from_abs(abs_path)  # (x,y) in letterbox pixels
                    except Exception:
                        center_xy_lb = None
                    center_xy_hm = None
                    if center_xy_lb is not None:
                        center_xy_hm = (
                            float(center_xy_lb[0]) / float(heatmap_stride_cfg),
                            float(center_xy_lb[1]) / float(heatmap_stride_cfg),
                        )

                    ts = ts_map.get(str(batch["image_path_abs"][bi]), None)
                    ts_str = str(ts) if ts is not None else ""
                    title = f"id={ev} | idx={int(ds_idx)}"
                    if ts_str:
                        title += f" | {ts_str}"
                    bottom_lines = [
                        f"gt_p={presence_gt:.2f}  pred_p={pred_prob:.2f}",
                        f"peak_logsumexp_topk={peak_logsumexp:.3f}",
                    ]

                    if pool is None:
                        tasks.append(
                            _render_three_panel_frame_cv2(
                                frame,
                                hm_tgt,
                                hm_pred_prob,
                                title=title,
                                gt_xy_hm=(float(tx), float(ty)),
                                pr_xy_hm=(float(px), float(py)),
                                center_xy_frame=(float(center_xy_lb[0]), float(center_xy_lb[1])) if center_xy_lb is not None else None,
                                center_xy_hm=center_xy_hm,
                                hm_pred_vmax=hm_pred_vmax,
                                bottom_lines=bottom_lines,
                                status_text=status_text,
                                status_color=status_color,
                                series_vals=peak_series,
                                series_index=series_idx,
                                series_label="",
                                series_marks=series_marks,
                            )
                        )
                    else:
                        tasks.append(
                            pool.submit(
                                _render_three_panel_frame_cv2,
                                frame,
                                hm_tgt,
                                hm_pred_prob,
                                title=title,
                                gt_xy_hm=(float(tx), float(ty)),
                                pr_xy_hm=(float(px), float(py)),
                                center_xy_frame=(float(center_xy_lb[0]), float(center_xy_lb[1])) if center_xy_lb is not None else None,
                                center_xy_hm=center_xy_hm,
                                hm_pred_vmax=hm_pred_vmax,
                                bottom_lines=bottom_lines,
                                status_text=status_text,
                                status_color=status_color,
                                series_vals=peak_series,
                                series_index=series_idx,
                                series_label="",
                                series_marks=series_marks,
                            )
                        )
                    frame_counter += 1

                for local_i, item in enumerate(tasks):
                    rgb = item if pool is None else item.result()
                    out_png = frames_dir / f"frame_{(frame_counter - len(tasks) + local_i):05d}.png"
                    try:
                        import cv2  # type: ignore
                        bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(out_png), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    except Exception:
                        plt.imsave(out_png, rgb)
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

        if _encode_with_ffmpeg(frames_dir, out_mp4):
            outputs[str(ev)] = out_mp4
            print(f"Saved (ffmpeg): {out_mp4}")
        else:
            cmd = f"ffmpeg -y -f concat -safe 0 -i {frames_dir}/frames.txt -framerate {int(fps)} -vsync vfr -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p {out_mp4}"
            print(f"[WARN] ffmpeg non eseguito automaticamente. Comando manuale: {cmd}")
            outputs[str(ev)] = frames_dir

    return outputs
