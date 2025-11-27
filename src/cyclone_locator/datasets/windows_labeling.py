"""Utilities for labeling frames from temporal windows manifests."""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LetterboxParams:
    src_w: int
    src_h: int
    target_size: int
    scale: float
    new_w: int
    new_h: int
    pad_x: int
    pad_y: int


@dataclass(frozen=True)
class Keypoint:
    x: float
    y: float
    source: Optional[str] = None
    cyclone_id: Optional[str] = None


@dataclass(frozen=True)
class EventTrack:
    event_id: Optional[str]
    start: pd.Timestamp
    end: pd.Timestamp
    timestamps: Tuple[pd.Timestamp, ...]
    keypoints: Tuple[Keypoint, ...]

    def contains(self, ts: pd.Timestamp) -> bool:
        return self.start <= ts <= self.end

    def keypoint_at(self, ts: pd.Timestamp) -> Optional[Keypoint]:
        if not self.contains(ts) or not self.timestamps:
            return None

        idx = bisect_left(self.timestamps, ts)
        if idx < len(self.timestamps) and self.timestamps[idx] == ts:
            return self.keypoints[idx]

        if idx <= 0:
            # Before first recorded point -> clamp to first
            base = self.keypoints[0]
            return base
        if idx >= len(self.timestamps):
            # After last recorded point -> clamp to last
            base = self.keypoints[-1]
            return base

        prev_idx = idx - 1
        next_idx = idx
        prev_ts = self.timestamps[prev_idx]
        next_ts = self.timestamps[next_idx]
        prev_kp = self.keypoints[prev_idx]
        next_kp = self.keypoints[next_idx]

        total = (next_ts - prev_ts).total_seconds()
        if total <= 0:
            return prev_kp
        alpha = (ts - prev_ts).total_seconds() / total
        x = prev_kp.x + alpha * (next_kp.x - prev_kp.x)
        y = prev_kp.y + alpha * (next_kp.y - prev_kp.y)
        cyclone_id = prev_kp.cyclone_id or next_kp.cyclone_id or self.event_id
        source = "interpolated"
        return Keypoint(float(x), float(y), source=source, cyclone_id=cyclone_id)


class WindowsLabeling:
    """Index temporal windows and optional keypoints for frame labeling."""

    def __init__(
        self,
        intervals: Sequence[Tuple[pd.Timestamp, pd.Timestamp]],
        timestamp_keypoints: Dict[pd.Timestamp, Keypoint],
        event_tracks: Sequence[EventTrack] | None = None,
    ) -> None:
        merged = self._merge_intervals(intervals)
        self._intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = merged
        self._keypoints = timestamp_keypoints
        self._event_tracks: Tuple[EventTrack, ...] = tuple(event_tracks) if event_tracks else tuple()

    @staticmethod
    def _merge_intervals(
        intervals: Sequence[Tuple[pd.Timestamp, pd.Timestamp]]
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        filtered = [iv for iv in intervals if iv[0] <= iv[1]]
        filtered.sort(key=lambda iv: iv[0])
        merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for start, end in filtered:
            if not merged:
                merged.append((start, end))
                continue
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    @classmethod
    @staticmethod
    def _detect_event_id_column(df: pd.DataFrame) -> Optional[str]:
        candidates = (
            "id_final",
            "id",
            "event_id",
            "cyclone_id",
            "id_cyc_unico",
            "idorig",
        )
        for col in candidates:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _normalize_event_id(value) -> Optional[str]:
        if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
            return None
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            fval = float(value)
            if fval.is_integer():
                return str(int(fval))
            return str(fval)
        text = str(value).strip()
        return text or None

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "WindowsLabeling":
        if "start_time" not in df.columns or "end_time" not in df.columns:
            raise ValueError("CSV must include start_time and end_time columns")
        intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        keypoints: Dict[pd.Timestamp, Keypoint] = {}
        event_tracks: Dict[str, Dict[str, object]] = {}
        seen_intervals = set()
        event_col = cls._detect_event_id_column(df)
        for _, row in df.iterrows():
            start = row.get("start_time")
            end = row.get("end_time")
            if isinstance(start, str) or isinstance(end, str):
                # pandas parse may leave str if parse_dates missing
                start = pd.to_datetime(start)
                end = pd.to_datetime(end)
            if pd.isna(start) or pd.isna(end):
                continue
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            key = (start_ts.to_datetime64(), end_ts.to_datetime64())
            if key not in seen_intervals:
                intervals.append((start_ts, end_ts))
                seen_intervals.add(key)

            event_id = cls._normalize_event_id(row.get(event_col)) if event_col else None
            if event_id is None:
                event_id = f"{start_ts.isoformat()}__{end_ts.isoformat()}"
            track = event_tracks.setdefault(
                event_id,
                {
                    "start": start_ts,
                    "end": end_ts,
                    "points": {},
                },
            )
            track["start"] = min(track["start"], start_ts)
            track["end"] = max(track["end"], end_ts)

            if "time" in row and not pd.isna(row["time"]):
                ts_value = row["time"]
                if isinstance(ts_value, str):
                    ts_value = pd.to_datetime(ts_value)
                timestamp = pd.Timestamp(ts_value)
                x = row["x_pix"] if "x_pix" in row.index else None
                y = row["y_pix"] if "y_pix" in row.index else None
                if x is not None and y is not None and not (pd.isna(x) or pd.isna(y)):
                    source = row.get("source") if isinstance(row, pd.Series) else None
                    cyclone_id = row.get("id_final") if isinstance(row, pd.Series) else None
                    cyclone_id = cls._normalize_event_id(cyclone_id) or event_id
                    kp_value = Keypoint(
                        float(x),
                        float(y),
                        source=str(source) if pd.notna(source) else None,
                        cyclone_id=str(cyclone_id) if cyclone_id is not None else None,
                    )
                    if timestamp not in keypoints:
                        keypoints[timestamp] = kp_value
                    track_points: Dict[pd.Timestamp, Keypoint] = track["points"]  # type: ignore[assignment]
                    track_points[timestamp] = kp_value

        track_objs: List[EventTrack] = []
        for event_id, data in event_tracks.items():
            point_dict: Dict[pd.Timestamp, Keypoint] = data["points"]  # type: ignore[assignment]
            sorted_items = sorted(point_dict.items(), key=lambda item: item[0])
            timestamps = tuple(ts for ts, _ in sorted_items)
            kp_values = tuple(kp for _, kp in sorted_items)
            track_objs.append(
                EventTrack(
                    event_id=event_id,
                    start=data["start"],  # type: ignore[arg-type]
                    end=data["end"],      # type: ignore[arg-type]
                    timestamps=timestamps,
                    keypoints=kp_values,
                )
            )

        return cls(intervals, keypoints, track_objs)

    @classmethod
    def from_csv(cls, csv_path: Path | str) -> "WindowsLabeling":
        df = pd.read_csv(csv_path, parse_dates=["time", "start_time", "end_time"], keep_default_na=True)
        return cls.from_dataframe(df)

    def is_positive(self, timestamp: datetime | pd.Timestamp) -> bool:
        ts = pd.Timestamp(timestamp)
        for start, end in self._intervals:
            if ts < start:
                break
            if start <= ts <= end:
                return True
        return False

    def keypoint_for(self, timestamp: datetime | pd.Timestamp) -> Optional[Keypoint]:
        ts = pd.Timestamp(timestamp)
        direct = self._keypoints.get(ts, None)
        if direct is not None:
            return direct
        for track in self._event_tracks:
            kp = track.keypoint_at(ts)
            if kp is not None:
                return kp
        return None

    def event_id_for(self, timestamp: datetime | pd.Timestamp) -> Optional[str]:
        """Return the event/cyclone id owning this timestamp, if any."""
        ts = pd.Timestamp(timestamp)
        for track in self._event_tracks:
            if track.contains(ts):
                return track.event_id
        return None

    def nearest_event(self, timestamp: datetime | pd.Timestamp):
        """Return (event_id, time_gap) of the closest event track to the timestamp."""
        ts = pd.Timestamp(timestamp)
        best_id: Optional[str] = None
        best_gap: Optional[pd.Timedelta] = None
        for track in self._event_tracks:
            if track.contains(ts):
                return track.event_id, pd.Timedelta(0)
            gap = min(abs(ts - track.start), abs(ts - track.end))
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_id = track.event_id
        return best_id, best_gap

    def has_keypoints(self) -> bool:
        return bool(self._keypoints)

    @property
    def intervals(self) -> Sequence[Tuple[pd.Timestamp, pd.Timestamp]]:
        return tuple(self._intervals)


def compute_letterbox_params(src_w: int, src_h: int, target_size: int) -> LetterboxParams:
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Source dimensions must be positive")
    if target_size <= 0:
        raise ValueError("Target size must be positive")
    scale = min(target_size / float(src_w), target_size / float(src_h))
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    pad_x = int(np.floor((target_size - new_w) / 2.0))
    pad_y = int(np.floor((target_size - new_h) / 2.0))
    return LetterboxParams(src_w, src_h, target_size, scale, new_w, new_h, pad_x, pad_y)


def project_keypoint(x: float, y: float, params: LetterboxParams) -> Tuple[int, int]:
    x_lb = int(round(x * params.scale)) + params.pad_x
    y_lb = int(round(y * params.scale)) + params.pad_y
    clipped_x = int(np.clip(x_lb, 0, params.target_size - 1))
    clipped_y = int(np.clip(y_lb, 0, params.target_size - 1))
    return clipped_x, clipped_y


def unproject_keypoint(x_lb: float, y_lb: float, params: LetterboxParams) -> Tuple[float, float]:
    x_orig = (x_lb - params.pad_x) / params.scale
    y_orig = (y_lb - params.pad_y) / params.scale
    return x_orig, y_orig


def parse_timestamp_from_filename(path: Path) -> Optional[pd.Timestamp]:
    stem = path.stem
    # Expect suffix like *_YYYYMMDD_HHMM
    tokens = stem.split("_")
    for tok_idx in range(len(tokens) - 1):
        date_token = tokens[tok_idx]
        time_token = tokens[tok_idx + 1]
        if len(date_token) == 8 and len(time_token) >= 4 and time_token[:4].isdigit() and date_token.isdigit():
            candidate = date_token + time_token[:4]
            try:
                dt = datetime.strptime(candidate, "%Y%m%d%H%M")
                return pd.Timestamp(dt)
            except ValueError:
                continue
    return None
