from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from cyclone_locator.datasets.temporal_utils import TemporalWindowSelector
from cyclone_locator.datasets.windows_labeling import parse_timestamp_from_filename


@dataclass(frozen=True)
class EventWindow:
    event_id: str
    start: pd.Timestamp
    end: pd.Timestamp


def load_event_windows_csv(windows_csv: str | Path) -> List[EventWindow]:
    """
    Parse medicanes_new_windows.csv and return unique event windows.
    Expected columns: start_time, end_time, and an id column (prefer id_final).
    """
    windows_csv = Path(windows_csv)
    df = pd.read_csv(windows_csv)
    if "start_time" not in df.columns or "end_time" not in df.columns:
        raise ValueError(f"{windows_csv} missing start_time/end_time columns")

    id_col = "id_final" if "id_final" in df.columns else ("id_cyc_unico" if "id_cyc_unico" in df.columns else None)
    if id_col is None:
        raise ValueError(f"{windows_csv} missing id_final/id_cyc_unico columns")

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
    df[id_col] = df[id_col].astype(str)

    events: List[EventWindow] = []
    grouped = df.dropna(subset=["start_time", "end_time"]).groupby(id_col, sort=True)
    for event_id, g in grouped:
        start = pd.Timestamp(g["start_time"].min())
        end = pd.Timestamp(g["end_time"].max())
        events.append(EventWindow(event_id=str(event_id), start=start, end=end))
    return events


def load_manifest_csv(manifest_csv: str | Path) -> pd.DataFrame:
    """
    Load manifest and normalize image_path to absolute paths (relative resolved from manifest directory).
    Requires columns: image_path, presence.
    """
    manifest_csv = Path(manifest_csv)
    df = pd.read_csv(manifest_csv)
    if "image_path" not in df.columns:
        raise ValueError(f"{manifest_csv} missing image_path column")
    if "presence" not in df.columns:
        raise ValueError(f"{manifest_csv} missing presence column")
    base = manifest_csv.parent

    def norm(p: str) -> str:
        p = str(p)
        if not p or p.lower() == "nan":
            return ""
        if os.path.isabs(p):
            return os.path.abspath(p)
        return os.path.abspath(str(base / p))

    df["image_path"] = df["image_path"].astype(str).apply(norm)
    df["presence"] = pd.to_numeric(df["presence"], errors="coerce").fillna(0.0).astype(float)
    df = df[df["image_path"] != ""].copy()
    exists = df["image_path"].apply(os.path.exists)
    df = df[exists].reset_index(drop=True)
    df["manifest_row"] = np.arange(len(df))
    df["timestamp"] = df["image_path"].apply(lambda p: parse_timestamp_from_filename(Path(p)) if p else None)
    return df


def _presence_probability_span(
    window_paths: List[str],
    selector: TemporalWindowSelector,
    presence_map: Dict[str, float],
    default_presence: float = 0.0,
) -> float:
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
        values.append(float(presence_map.get(abs_path, default_presence)))
    if not values:
        return float(default_presence)
    return float(np.mean(values))


def compute_presence_gt_prob(
    df_full: pd.DataFrame,
    temporal_T: int,
    temporal_stride: int,
    *,
    manifest_stride: int = 1,
) -> pd.DataFrame:
    """
    Compute presence_gt_prob for samples as done in MedFullBasinDataset:
    - Use TemporalWindowSelector(window) for each center frame
    - Take span from earliest to latest frame in window
    - Return mean presence over that span.

    Returns a dataframe containing only samples that would be seen by the dataset
    given manifest_stride, with columns:
      - dataset_idx, manifest_row (original row), image_path, timestamp, presence, presence_gt_prob
    """
    temporal_T = max(1, int(temporal_T))
    temporal_stride = max(1, int(temporal_stride))
    manifest_stride = max(1, int(manifest_stride))

    selector = TemporalWindowSelector(temporal_T=temporal_T, temporal_stride=temporal_stride)
    presence_map = {os.path.abspath(p): float(v) for p, v in zip(df_full["image_path"], df_full["presence"])}

    # dataset sees only every manifest_stride row (iloc[::stride])
    kept = df_full.iloc[::manifest_stride].copy().reset_index(drop=False)
    kept = kept.rename(columns={"index": "manifest_row_orig"})
    kept["dataset_idx"] = np.arange(len(kept))

    probs = []
    for p in kept["image_path"].tolist():
        w = selector.get_window(p)
        probs.append(_presence_probability_span(w, selector, presence_map, default_presence=0.0))
    kept["presence_gt_prob"] = np.asarray(probs, dtype=float)
    return kept


def _pick_nearest_by_time(df: pd.DataFrame, target: pd.Timestamp) -> Optional[pd.Series]:
    ts = df["timestamp"]
    if ts.isna().all():
        return None
    dt = (ts - target).abs()
    idx = dt.idxmin()
    return df.loc[idx]


def _pick_transition(df: pd.DataFrame, target: pd.Timestamp, hours: float = 12.0) -> Optional[pd.Series]:
    if df.empty:
        return None
    ts = df["timestamp"]
    if ts.isna().all():
        return None
    lo = target - pd.Timedelta(hours=hours)
    hi = target + pd.Timedelta(hours=hours)
    cand = df[(ts >= lo) & (ts <= hi)]
    if cand.empty:
        return None
    # Prefer closest to 0.5, then closest to time target
    score = (cand["presence_gt_prob"] - 0.5).abs()
    time_dist = (cand["timestamp"] - target).abs()
    order = np.lexsort((time_dist.to_numpy(), score.to_numpy()))
    return cand.iloc[int(order[0])]


def representative_event_samples(
    manifest_csv: str | Path,
    windows_csv: str | Path,
    *,
    temporal_T: int,
    temporal_stride: int,
    manifest_stride: int = 1,
) -> pd.DataFrame:
    """
    For each event present in the manifest time range, pick 5 representative dataset indices:
      - before_start
      - transition (presence_gt_prob in (0,1) near start or end)
      - middle (near midpoint)
      - final (near end but inside)
      - after_end

    Returns a table with one row per (event, phase).
    """
    events = load_event_windows_csv(windows_csv)
    df_full = load_manifest_csv(manifest_csv)
    df_ds = compute_presence_gt_prob(
        df_full,
        temporal_T=temporal_T,
        temporal_stride=temporal_stride,
        manifest_stride=manifest_stride,
    )

    # Work on dataset-visible subset; drop rows without timestamp
    df_ds = df_ds.copy()
    df_ds["timestamp"] = df_ds["timestamp"].astype("datetime64[ns]")

    rows: List[Dict[str, object]] = []
    for ev in events:
        ts = df_ds["timestamp"]
        in_ev = (ts >= ev.start) & (ts <= ev.end)
        if not bool(in_ev.any()):
            continue

        inside = df_ds[in_ev]
        # Use the dominant directory for this event to pick "before/after" from the same sequence.
        inside_dirs = inside["image_path"].apply(lambda p: os.path.dirname(str(p))).astype(str)
        if inside_dirs.empty:
            continue
        dom_dir = inside_dirs.value_counts().idxmax()
        same_dir = df_ds[df_ds["image_path"].apply(lambda p: os.path.dirname(str(p)) == dom_dir)]

        before = same_dir[same_dir["timestamp"] < ev.start].tail(1)
        after = same_dir[same_dir["timestamp"] > ev.end].head(1)
        inside = inside[inside["image_path"].apply(lambda p: os.path.dirname(str(p)) == dom_dir)]
        end_row = inside.tail(1)
        mid_t = ev.start + (ev.end - ev.start) / 2
        mid_row = _pick_nearest_by_time(inside, mid_t)

        trans_pool = same_dir[(same_dir["presence_gt_prob"] > 1e-6) & (same_dir["presence_gt_prob"] < 1 - 1e-6)]
        trans = _pick_transition(trans_pool, ev.start)
        if trans is None:
            trans = _pick_transition(trans_pool, ev.end)

        picks: List[Tuple[str, Optional[pd.Series]]] = [
            ("before_start", before.iloc[0] if not before.empty else None),
            ("transition", trans),
            ("middle", mid_row),
            ("final", end_row.iloc[0] if not end_row.empty else None),
            ("after_end", after.iloc[0] if not after.empty else None),
        ]

        for phase, r in picks:
            if r is None:
                continue
            rows.append(
                {
                    "event_id": ev.event_id,
                    "phase": phase,
                    "dataset_idx": int(r["dataset_idx"]),
                    "manifest_row": int(r["manifest_row_orig"]),
                    "timestamp": r["timestamp"],
                    "presence_frame": float(r["presence"]),
                    "presence_gt_prob": float(r["presence_gt_prob"]),
                    "image_path": str(r["image_path"]),
                    "event_start": ev.start,
                    "event_end": ev.end,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Stable ordering
    out = out.sort_values(["event_id", "phase"]).reset_index(drop=True)
    return out
