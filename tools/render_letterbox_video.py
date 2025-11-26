#!/usr/bin/env python3
"""
Render a single MP4 with predicted vs GT cyclone centers on letterboxed frames.

This CLI:
- joins preds.csv and manifest.csv on the image path (letterbox SxS),
- aligns timestamps, splits into contiguous segments via a gap threshold,
- draws red (pred) / green (GT) dots on each frame PNG,
- delegates frames.txt + ffmpeg encoding to the external rendering utilities
  copied from the VideoMAE "View MED tracking preds" tooling.

If the external rendering helpers are missing, the script fails with a clear
message instructing to copy them.
"""
from __future__ import annotations

import argparse
import inspect
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _cand in (REPO_ROOT, HERE):
    _cand_str = str(_cand)
    if _cand_str not in sys.path:
        sys.path.insert(0, _cand_str)

try:
    from view_test_tiles import make_animation_parallel_ffmpeg  # type: ignore
    _RENDER_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:  # pragma: no cover
    make_animation_parallel_ffmpeg = None  # type: ignore
    _RENDER_IMPORT_ERROR = e
    raise RuntimeError(
        "Funzione di rendering non disponibile — copiare i tool di View MED nel path (view_test_tiles.make_animation_parallel_ffmpeg)."
    ) from _RENDER_IMPORT_ERROR


DEFAULT_PATH_CANDIDATES = ("resized_path", "image_path", "path")
DEFAULT_TIME_REGEX = r"\d{4}[-_]?(\d{2})[-_]?(\d{2})[T _-]?(\d{2})[-_:]?(\d{2})"
DEFAULT_FILENAME_FORMATS = (
    "yyyy-MM-ddTHH-mm",
    "yyyy-MM-dd_HH-mm",
    "yyyyMMdd_HHmm",
    "yyyyMMddHHmm",
)
PY_STRPTIME_MAP = {
    "yyyy": "%Y",
    "MM": "%m",
    "dd": "%d",
    "HH": "%H",
    "mm": "%M",
}
PRED_X_CANDIDATES = ("x_g", "pred_x", "track_pred_x", "x")
PRED_Y_CANDIDATES = ("y_g", "pred_y", "track_pred_y", "y")
GT_X_CANDIDATES = ("x_pix_resized", "cx", "x_gt")
GT_Y_CANDIDATES = ("y_pix_resized", "cy", "y_gt")
PRESENCE_PROB_CANDIDATES = ("presence_prob", "prob", "presence_pred")


def to_strptime(fmt: str) -> str:
    out = fmt
    for k, v in PY_STRPTIME_MAP.items():
        out = out.replace(k, v)
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render letterbox prediction video (pred vs GT).")
    p.add_argument("--preds_csv", required=True, type=Path, help="CSV with predictions in letterbox space.")
    p.add_argument("--manifest_csv", required=True, type=Path, help="Test manifest CSV with presence/GT columns.")
    p.add_argument("--out_mp4", required=True, type=Path, help="Output MP4 path.")
    p.add_argument("--frames_dir", required=True, type=Path, help="Folder for intermediate PNG frames.")
    p.add_argument("--fps", type=int, default=12, help="Playback FPS for the video.")
    p.add_argument("--gap-minutes", type=float, default=600.0, help="Start new segment when delta minutes exceed this.")
    p.add_argument("--segment-slate-seconds", type=float, default=1.0, help="Seconds of slate/black between segments.")
    p.add_argument("--image-size", type=int, default=384, help="Letterbox side S (pixels).")
    return p


def normalize_path(val) -> Optional[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    # Normalize separators only; keep case to stay case-sensitive where needed.
    return s.replace("\\", "/")


def pick_path_column(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    for cand in DEFAULT_PATH_CANDIDATES:
        if cand in df.columns:
            logging.info("Column '%s' missing; auto-selecting '%s' for paths.", requested, cand)
            return cand
    raise ValueError(f"None of the path columns { [requested] + list(DEFAULT_PATH_CANDIDATES) } exist in CSV.")


def pick_column(df: pd.DataFrame, candidates: Sequence[str], kind: str) -> str:
    for cand in candidates:
        if cand in df.columns:
            logging.info("Using column '%s' for %s.", cand, kind)
            return cand
    raise ValueError(f"Missing required column for {kind}. Tried: {candidates}")


def parse_dt_flex(text: str, formats: Sequence[str]) -> Optional[datetime]:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    text = str(text).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except Exception:
        pass
    for f in formats:
        try:
            return datetime.strptime(text, to_strptime(f))
        except Exception:
            continue
    return None


def parse_dt_from_filename(path: str, time_re: re.Pattern, formats: Sequence[str]) -> Optional[datetime]:
    name = Path(path).name
    m = time_re.search(name)
    if not m:
        return None
    raw = m.group(0)
    dt = parse_dt_flex(raw, formats)
    if dt is not None:
        return dt
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 12:
        try:
            return datetime.strptime(digits, "%Y%m%d%H%M")
        except Exception:
            return None
    return None


def assign_timestamps(
    df: pd.DataFrame, path_col: str, time_re: re.Pattern, formats: Sequence[str]
) -> pd.Series:
    datetime_cols = [c for c in ("datetime_iso", "datetime", "frame_datetime") if c in df.columns]
    if datetime_cols:
        col = datetime_cols[0]
        logging.info("Using datetime column '%s' when available; fallback to filename parsing if empty.", col)
        parsed = df[col].apply(lambda x: parse_dt_flex(x, formats))
        missing_mask = parsed.isna()
        if missing_mask.any():
            fallback = df.loc[missing_mask, path_col].apply(lambda p: parse_dt_from_filename(p, time_re, formats))
            parsed.loc[missing_mask] = fallback
        return parsed
    logging.info("No datetime column found; parsing timestamp from filename with regex %s.", time_re.pattern)
    return df[path_col].apply(lambda p: parse_dt_from_filename(p, time_re, formats))


def assign_segments(timestamps: Iterable[datetime], gap_minutes: float) -> List[int]:
    seg_ids: List[int] = []
    current_seg = 0
    prev_ts: Optional[datetime] = None
    for ts in timestamps:
        if prev_ts is not None and ts is not None:
            delta = (ts - prev_ts).total_seconds() / 60.0
            if delta > gap_minutes:
                current_seg += 1
        seg_ids.append(current_seg)
        prev_ts = ts
    return seg_ids


def clamp_point(x: Optional[float], y: Optional[float], size: int) -> Optional[Tuple[int, int]]:
    if x is None or y is None:
        return None
    if isinstance(x, float) and pd.isna(x):
        return None
    if isinstance(y, float) and pd.isna(y):
        return None
    try:
        xi = int(np.clip(float(x), 0, size - 1))
        yi = int(np.clip(float(y), 0, size - 1))
        return (xi, yi)
    except Exception:
        return None


def draw_point(img: np.ndarray, center: Tuple[int, int], color: Tuple[int, int, int], radius: int = 4) -> None:
    # Outer black outline for visibility
    cv2.circle(img, center, radius + 1, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(img, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)


def render_frame(
    image_path: Path,
    out_path: Path,
    pred_xy: Optional[Tuple[int, int]],
    gt_xy: Optional[Tuple[int, int]],
    image_size: int,
    ts: Optional[datetime],
    presence_prob: Optional[float],
) -> bool:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        logging.warning("Skipping missing or unreadable image: %s", image_path)
        return False
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if h != image_size or w != image_size:
        logging.warning(
            "Image size mismatch for %s: got %sx%s, expected %s. Proceeding without resize.", image_path, h, w, image_size
        )
    if pred_xy is not None:
        draw_point(img, pred_xy, (0, 0, 255))
    if gt_xy is not None:
        draw_point(img, gt_xy, (0, 255, 0))

    label_parts: List[str] = []
    if ts is not None:
        label_parts.append(ts.strftime("%Y-%m-%d %H:%M"))
    if presence_prob is not None and not pd.isna(presence_prob):
        try:
            label_parts.append(f"prob: {float(presence_prob):.2f}")
        except Exception:
            pass
    if label_parts:
        text = " | ".join(label_parts)
        text_xy = (8, 18)
        cv2.putText(
            img,
            text,
            text_xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img,
            text,
            text_xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), img)
    if not ok:
        logging.warning("Failed to write frame PNG: %s", out_path)
    return bool(ok)


def build_slate_frames(
    frames_dir: Path, image_size: int, slate_seconds: float, fps: int, start_idx: int, segment_id: int
) -> List[dict]:
    if slate_seconds <= 0:
        return []
    num_frames = int(round(slate_seconds * fps))
    if num_frames <= 0:
        return []
    slate = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    slate_path = frames_dir / f"slate_segment{segment_id:03d}_{start_idx:06d}.png"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(slate_path), slate)
    return [
        {
            "path": str(slate_path),
            "timestamp": None,
            "segment_id": segment_id,
            "is_slate": True,
        }
        for _ in range(num_frames)
    ]


def call_external_renderer(frames: pd.DataFrame, frames_dir: Path, out_mp4: Path, fps: int) -> None:
    sig = inspect.signature(make_animation_parallel_ffmpeg)
    kwargs = {}
    if "output_folder" in sig.parameters:
        kwargs["output_folder"] = str(frames_dir)
    if "nomefile" in sig.parameters:
        kwargs["nomefile"] = out_mp4.stem
    if "fps" in sig.parameters:
        kwargs["fps"] = fps
    if "out_path" in sig.parameters:
        kwargs["out_path"] = str(out_mp4)
    if "frames_txt_path" in sig.parameters:
        kwargs["frames_txt_path"] = str(out_mp4.with_suffix(".txt"))

    logging.info(
        "Invoking make_animation_parallel_ffmpeg with output_folder=%s, nomefile=%s, fps=%s",
        kwargs.get("output_folder", str(frames_dir)),
        kwargs.get("nomefile", out_mp4.stem),
        kwargs.get("fps", fps),
    )
    try:
        make_animation_parallel_ffmpeg(frames, **kwargs)
    except TypeError as e:
        raise RuntimeError(
            "make_animation_parallel_ffmpeg signature mismatch. Copiare la versione prevista dal notebook VideoMAE oppure aggiornare l'adapter."
        ) from e


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    preds_df = pd.read_csv(args.preds_csv)
    manifest_df = pd.read_csv(args.manifest_csv)

    preds_path_col = pick_path_column(preds_df, "resized_path")
    manifest_path_col = pick_path_column(manifest_df, "resized_path")

    preds_df["_path_norm"] = preds_df[preds_path_col].apply(normalize_path)
    manifest_df["_path_norm"] = manifest_df[manifest_path_col].apply(normalize_path)

    preds_df["_image_path"] = preds_df[preds_path_col]
    manifest_df["_image_path"] = manifest_df[manifest_path_col]
    time_re = re.compile(DEFAULT_TIME_REGEX)
    preds_df["_timestamp"] = assign_timestamps(preds_df, preds_path_col, time_re, DEFAULT_FILENAME_FORMATS)
    manifest_df["_timestamp"] = assign_timestamps(manifest_df, manifest_path_col, time_re, DEFAULT_FILENAME_FORMATS)

    merged = preds_df.merge(manifest_df, on="_path_norm", how="inner", suffixes=("_pred", "_gt"))
    if merged.empty:
        raise RuntimeError("Join between preds and manifest is empty. Check path columns and normalization.")

    merged["_timestamp"] = merged.pop("_timestamp_pred").combine_first(merged.pop("_timestamp_gt"))
    merged["_image_path"] = merged.pop("_image_path_gt").combine_first(merged.pop("_image_path_pred"))
    merged = merged[merged["_timestamp"].notna()].copy()
    merged = merged.sort_values("_timestamp").reset_index(drop=True)
    merged["segment_id"] = assign_segments(merged["_timestamp"].tolist(), args.gap_minutes)

    pred_x_col = pick_column(merged, PRED_X_CANDIDATES, "pred x")
    pred_y_col = pick_column(merged, PRED_Y_CANDIDATES, "pred y")

    gt_x_col = next((c for c in GT_X_CANDIDATES if c in merged.columns), None)
    gt_y_col = next((c for c in GT_Y_CANDIDATES if c in merged.columns), None)
    missing_gt = not (gt_x_col and gt_y_col)
    if not missing_gt:
        if merged[gt_x_col].isna().all() or merged[gt_y_col].isna().all():
            missing_gt = True
    if missing_gt:
        logging.warning("GT center columns not found or all NaN; only prediction dots will be drawn.")

    presence_prob_col = next((c for c in PRESENCE_PROB_CANDIDATES if c in merged.columns), None)

    logging.info(
        "Loaded %d preds rows, %d manifest rows; joined %d rows with timestamps.",
        len(preds_df),
        len(manifest_df),
        len(merged),
    )

    frames_dir = args.frames_dir
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_records: List[dict] = []
    global_idx = 0
    unique_segments = merged["segment_id"].unique().tolist()
    unique_segments.sort()
    for i, seg_id in enumerate(unique_segments):
        seg = merged[merged["segment_id"] == seg_id]
        for _, row in seg.iterrows():
            pred_xy = clamp_point(row.get(pred_x_col), row.get(pred_y_col), args.image_size)
            gt_xy = clamp_point(row.get(gt_x_col), row.get(gt_y_col), args.image_size) if not missing_gt else None
            out_path = frames_dir / f"frame_{global_idx:06d}.png"
            ok = render_frame(
                Path(row["_image_path"]),
                out_path,
                pred_xy,
                gt_xy,
                args.image_size,
                row["_timestamp"],
                row.get(presence_prob_col) if presence_prob_col else None,
            )
            if ok:
                frame_records.append(
                    {
                        "path": str(out_path),
                        "timestamp": row["_timestamp"],
                        "segment_id": seg_id,
                        "is_slate": False,
                    }
                )
                global_idx += 1

        is_last_segment = i == len(unique_segments) - 1
        if not is_last_segment and args.segment_slate_seconds > 0 and frame_records:
            slate_frames = build_slate_frames(
                frames_dir, args.image_size, args.segment_slate_seconds, args.fps, global_idx, seg_id
            )
            frame_records.extend(slate_frames)
            global_idx += len(slate_frames)

    if not frame_records:
        raise RuntimeError("No frames were written; check inputs and columns.")

    frames_df = pd.DataFrame(frame_records)
    logging.info(
        "Prepared %d PNG frames across %d segments. Handing off to external renderer.",
        len(frames_df),
        frames_df['segment_id'].nunique(),
    )

    call_external_renderer(frames_df, frames_dir, args.out_mp4, args.fps)
    logging.info("Done. MP4 should be at: %s", args.out_mp4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
