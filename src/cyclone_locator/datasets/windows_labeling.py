"""Utilities for labeling frames from temporal windows manifests."""
from __future__ import annotations

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


class WindowsLabeling:
    """Index temporal windows and optional keypoints for frame labeling."""

    def __init__(
        self,
        intervals: Sequence[Tuple[pd.Timestamp, pd.Timestamp]],
        timestamp_keypoints: Dict[pd.Timestamp, Keypoint],
    ) -> None:
        merged = self._merge_intervals(intervals)
        self._intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = merged
        self._keypoints = timestamp_keypoints

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
    def from_dataframe(cls, df: pd.DataFrame) -> "WindowsLabeling":
        if "start_time" not in df.columns or "end_time" not in df.columns:
            raise ValueError("CSV must include start_time and end_time columns")
        intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        keypoints: Dict[pd.Timestamp, Keypoint] = {}
        seen_intervals = set()
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

            if "time" in row and not pd.isna(row["time"]):
                ts_value = row["time"]
                if isinstance(ts_value, str):
                    ts_value = pd.to_datetime(ts_value)
                timestamp = pd.Timestamp(ts_value)
                x = row["x_pix"] if "x_pix" in row.index else None
                y = row["y_pix"] if "y_pix" in row.index else None
                if x is not None and y is not None and not (pd.isna(x) or pd.isna(y)):
                    if timestamp not in keypoints:
                        source = row.get("source") if isinstance(row, pd.Series) else None
                        cyclone_id = row.get("id_final") if isinstance(row, pd.Series) else None
                        keypoints[timestamp] = Keypoint(float(x), float(y),
                                                        source=str(source) if pd.notna(source) else None,
                                                        cyclone_id=str(cyclone_id) if pd.notna(cyclone_id) else None)
        return cls(intervals, keypoints)

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
        return self._keypoints.get(ts, None)

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
