#!/usr/bin/env python3
"""Render ROI videos for all events in train/val/test manifests."""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
SRC = ROOT / "src"
NOTEBOOKS = REPO_ROOT / "notebooks"

if SRC.exists():
    import sys

    sys.path.insert(0, str(SRC))
    sys.path.insert(0, str(NOTEBOOKS))

from cyclone_locator.datasets.med_fullbasin import MedFullBasinDataset  # noqa: E402
from cyclone_locator.datasets.temporal_utils import TemporalWindowSelector  # noqa: E402
from cyclone_locator.infer import build_model, load_letterbox_meta, spatial_peak_pool  # noqa: E402
from cyclone_locator.utils.metric import peak_and_width  # noqa: E402

from firstpass_videomae_roi_viz_utils import (  # noqa: E402
    build_manifest_map,
    build_meta_by_name,
    render_event_video,
)


def _find_ffmpeg(preferred: Optional[str] = None) -> Optional[str]:
    if preferred and os.path.exists(preferred) and os.access(preferred, os.X_OK):
        return preferred
    for env_key in ("FFMPEG", "IMAGEIO_FFMPEG_EXE"):
        v = os.environ.get(env_key)
        if v and os.path.exists(v) and os.access(v, os.X_OK):
            return v
    local_static = "/media/fenrir/disk1/danieleda/ffmpeg-7.0.2-amd64-static/ffmpeg"
    if os.path.exists(local_static) and os.access(local_static, os.X_OK):
        return local_static
    return shutil.which("ffmpeg")


def encode_ffmpeg(frames_dir: Path, out_mp4: Path, fps: int, ffmpeg_path: Optional[str] = None) -> Optional[Path]:
    frames = sorted(Path(frames_dir).glob("frame_*.png"))
    if not frames:
        return None
    frames_txt = Path(frames_dir) / "frames.txt"
    dt = 1.0 / float(max(1, int(fps)))
    with open(frames_txt, "w", encoding="utf-8") as f:
        for p in frames[:-1]:
            f.write(f"file '{p.resolve()}'\n")
            f.write(f"duration {dt:.9f}\n")
        last = frames[-1].resolve()
        f.write(f"file '{last}'\n")
    ffmpeg = _find_ffmpeg(ffmpeg_path)
    if ffmpeg is None:
        print("ffmpeg not found. Use frames_dir and frames.txt to encode manually.")
        return None
    cmd = [
        ffmpeg,
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
        import subprocess

        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        print(exc.stderr)
        raise
    return out_mp4


def _abs_path_from_manifest(manifest_csv: Path, path_str: str) -> str:
    if os.path.isabs(path_str):
        return os.path.abspath(path_str)
    return os.path.abspath(os.path.join(str(manifest_csv.parent), path_str))


def _event_ids_from_df(df: pd.DataFrame) -> List[str]:
    series = df["event_id"].dropna().astype(str)
    return sorted(series.unique().tolist())


def _iter_splits(splits: Sequence[str]) -> Iterable[str]:
    for s in splits:
        s = s.strip().lower()
        if s:
            yield s


def main() -> None:
    # --------- CONFIG (edit these paths/values) ----------
    RUN_DIR = Path("/media/fenrir/disk1/danieleda/Demetra/moduli/firstpass/outputs/runs/exp_33_energy-presence")
    CHECKPOINT = RUN_DIR / "best.ckpt"
    CONFIG_YAML = RUN_DIR / "config_used.yml"
    OUT_DIR = REPO_ROOT / "notebooks" / "roi_viz_out_all"
    SPLITS = ("train", "val", "test")
    MANIFEST_TRAIN = None  # override if needed
    MANIFEST_VAL = None
    MANIFEST_TEST = None
    LETTERBOX_META = None
    EVENT_IDS = None  # e.g. ["46218", "7001500"]
    MAX_EVENTS = None  # e.g. 10
    BATCH_SIZE = None
    NUM_WORKERS = 8
    RENDER_WORKERS = 8
    PRESENCE_THRESHOLD = None
    FPS = 6
    ENCODE_MP4 = True
    USE_SOFT_ARGMAX = True
    USE_ENERGY_FUSION = True
    TEMPORAL_T = None
    TEMPORAL_STRIDE = None
    MANIFEST_STRIDE = None
    FFMPEG_PATH = "/media/fenrir/disk1/danieleda/ffmpeg-7.0.2-amd64-static/ffmpeg"
    # -----------------------------------------------------

    cfg_path = CONFIG_YAML
    if cfg_path is None or not cfg_path.exists():
        cfg_path = ROOT / "config" / "default.yml"

    cfg = yaml.safe_load(open(cfg_path))

    if MANIFEST_TRAIN is None:
        MANIFEST_TRAIN = ROOT / cfg["data"]["manifest_train"]
    if MANIFEST_VAL is None:
        MANIFEST_VAL = ROOT / cfg["data"]["manifest_val"]
    if MANIFEST_TEST is None:
        MANIFEST_TEST = ROOT / cfg["data"]["manifest_test"]
    if LETTERBOX_META is None and cfg["data"].get("letterbox_meta_csv"):
        LETTERBOX_META = ROOT / cfg["data"]["letterbox_meta_csv"]

    image_size = int(cfg.get("train", {}).get("image_size", 224))
    heatmap_stride = int(cfg.get("train", {}).get("heatmap_stride", 2))
    heatmap_sigma_px = float(cfg.get("loss", {}).get("heatmap_sigma_px", 8.0))
    manifest_stride = int(MANIFEST_STRIDE or cfg.get("data", {}).get("manifest_stride", 1) or 1)
    use_pre_letterboxed = bool(cfg.get("data", {}).get("use_pre_letterboxed", True))
    letterbox_size_assert = cfg.get("data", {}).get("letterbox_size_assert", None)

    temporal_T = int(TEMPORAL_T or cfg.get("train", {}).get("temporal_T", 1) or 1)
    temporal_stride = int(TEMPORAL_STRIDE or cfg.get("train", {}).get("temporal_stride", 1) or 1)

    roi_base_radius = int(cfg.get("infer", {}).get("roi_base_radius_px", 112))
    roi_sigma_multiplier = float(cfg.get("infer", {}).get("roi_sigma_multiplier", 2.0))

    loss_cfg = cfg.get("loss", {})
    infer_cfg = cfg.get("infer", {})
    peak_pool = str(loss_cfg.get("peak_pool", "max"))
    peak_tau = float(loss_cfg.get("peak_tau", 1.0))
    presence_threshold = float(PRESENCE_THRESHOLD) if PRESENCE_THRESHOLD is not None else float(
        infer_cfg.get("peak_threshold") or infer_cfg.get("presence_threshold") or 0.5
    )
    presence_from_peak = bool(
        cfg.get("infer", {}).get("presence_from_peak", cfg.get("train", {}).get("presence_from_peak", False))
    )
    presence_topk_cfg = int(cfg.get("train", {}).get("presence_topk", 0) or 0)
    presence_topk = presence_topk_cfg if presence_topk_cfg > 0 else None
    batch_size = int(BATCH_SIZE or infer_cfg.get("batch_size") or 16)

    center_tau = float(cfg.get("infer", {}).get("center_tau", cfg.get("loss", {}).get("dsnt_tau", 1.0)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger("render_all_events")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())

    model = build_model(
        cfg,
        str(CHECKPOINT),
        device,
        logger,
        temporal_T=temporal_T,
        heatmap_stride=heatmap_stride,
        use_energy_fusion=bool(USE_ENERGY_FUSION),
    )

    selector = TemporalWindowSelector(temporal_T, temporal_stride)
    meta_map = load_letterbox_meta(str(LETTERBOX_META)) if LETTERBOX_META else {}
    meta_by_name = build_meta_by_name(meta_map)

    event_filter = {str(e).strip() for e in EVENT_IDS} if EVENT_IDS else None

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = {
        "train": MANIFEST_TRAIN,
        "val": MANIFEST_VAL,
        "test": MANIFEST_TEST,
    }

    for split in _iter_splits(SPLITS):
        manifest_csv = manifest_paths.get(split)
        if not manifest_csv:
            continue
        if not manifest_csv.exists():
            raise FileNotFoundError(manifest_csv)
        print(f"[{split}] manifest={manifest_csv}")

        ds = MedFullBasinDataset(
            csv_path=str(manifest_csv),
            image_size=image_size,
            heatmap_stride=heatmap_stride,
            heatmap_sigma_px=heatmap_sigma_px,
            use_aug=False,
            use_pre_letterboxed=use_pre_letterboxed,
            letterbox_meta_csv=str(LETTERBOX_META) if LETTERBOX_META else None,
            letterbox_size_assert=letterbox_size_assert,
            temporal_T=temporal_T,
            temporal_stride=temporal_stride,
            manifest_stride=manifest_stride,
        )

        manifest_df = pd.read_csv(manifest_csv)
        manifest_df["image_path_abs"] = manifest_df["image_path"].apply(
            lambda p: _abs_path_from_manifest(manifest_csv, str(p))
        )
        manifest_map = build_manifest_map(manifest_df)

        event_ids = _event_ids_from_df(ds.df)
        if event_filter is not None:
            event_ids = [eid for eid in event_ids if eid in event_filter]
        if MAX_EVENTS is not None:
            event_ids = event_ids[: int(MAX_EVENTS)]

        split_out = out_dir / split
        split_out.mkdir(parents=True, exist_ok=True)

        for event_id in event_ids:
            sub = ds.df[ds.df["event_id"].astype(str) == str(event_id)]
            if sub.empty:
                continue
            event_indices = [int(i) for i in sub.index.tolist()]
            output_name = f"{split}_event_{event_id}"
            print(f"[{split}] event_id={event_id} frames={len(event_indices)}")
            frames_dir, _ = render_event_video(
                ds=ds,
                event_indices=event_indices,
                model=model,
                device=device,
                meta_map=meta_map,
                meta_by_name=meta_by_name,
                manifest_map=manifest_map,
                selector=selector,
                heatmap_stride=heatmap_stride,
                center_tau=center_tau,
                roi_base_radius=roi_base_radius,
                roi_sigma_multiplier=roi_sigma_multiplier,
                presence_threshold=presence_threshold,
                peak_pool=peak_pool,
                peak_tau=peak_tau,
                presence_topk=presence_topk,
                presence_from_peak=presence_from_peak,
                batch_size=batch_size,
                num_workers=int(NUM_WORKERS),
                render_workers=int(RENDER_WORKERS),
                output_dir=split_out,
                output_name=output_name,
                event_id=str(event_id),
                use_soft_argmax=bool(USE_SOFT_ARGMAX),
            )
            if ENCODE_MP4:
                out_mp4 = split_out / f"{output_name}.mp4"
                encode_ffmpeg(Path(frames_dir), out_mp4, int(FPS), FFMPEG_PATH)


if __name__ == "__main__":
    main()
