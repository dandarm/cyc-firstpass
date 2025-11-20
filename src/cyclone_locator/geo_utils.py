"""Wrapper attorno alle funzioni di medicane_utils per conversioni pixel↔geo."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

try:
    from medicane_utils.geo_const import get_lon_lat_grid_2_pixel
except Exception:  # pragma: no cover - dependency not available
    get_lon_lat_grid_2_pixel = None  # type: ignore[assignment]

IMAGE_WIDTH = 1290
IMAGE_HEIGHT = 420
EARTH_RADIUS_KM = 6371.0088


def _ensure_geo_backend() -> None:
    if get_lon_lat_grid_2_pixel is None:
        raise RuntimeError(
            "medicane_utils.geo_const non disponibile: installa basemap e assicurati che "
            "il pacchetto medicane_utils sia nel PYTHONPATH."
        )


@lru_cache(maxsize=1)
def _get_lon_lat_grid() -> Tuple[np.ndarray, np.ndarray]:
    _ensure_geo_backend()
    lon_grid, lat_grid, _, _ = get_lon_lat_grid_2_pixel(IMAGE_WIDTH, IMAGE_HEIGHT)  # type: ignore[misc]
    return lon_grid, lat_grid


def pixels_to_latlon(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lon_grid, lat_grid = _get_lon_lat_grid()
    x_idx = np.clip(np.rint(x).astype(int), 0, IMAGE_WIDTH - 1)
    y_idx = np.clip(np.rint(y).astype(int), 0, IMAGE_HEIGHT - 1)
    row_idx = IMAGE_HEIGHT - 1 - y_idx
    lat = lat_grid[row_idx, x_idx]
    lon = lon_grid[row_idx, x_idx]
    return lat, lon


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c
