import os
from typing import Dict, List


class TemporalWindowSelector:
    """Selects a centered temporal window of file paths.

    The selector keeps a cached, sorted listing per directory to avoid
    repeatedly touching the filesystem when workers spawn. The window size is
    defined by ``temporal_T`` and the spacing by ``temporal_stride``; for
    ``T=1`` the original path is returned.

    Notes:
    - For odd ``temporal_T`` the window is symmetric around the center index.
    - For even ``temporal_T`` there is no unique "center" frame; we use the
      convention offsets ``[-T//2, ..., -1, 0, ..., T//2-1]`` so the center
      frame is included and becomes the second of the two middle frames.
    """

    def __init__(self, temporal_T: int = 1, temporal_stride: int = 1):
        self.temporal_T = max(1, int(temporal_T))
        self.temporal_stride = max(1, int(temporal_stride))
        self.half = self.temporal_T // 2
        self._dir_cache: Dict[str, List[str]] = {}
        self._dir_index: Dict[str, Dict[str, int]] = {}

    def offsets(self) -> List[int]:
        """Returns the list of integer offsets used to build the temporal window."""
        return self._offsets()

    def _offsets(self) -> List[int]:
        if self.temporal_T == 1:
            return [0]
        if self.temporal_T % 2 == 1:
            return list(range(-self.half, self.half + 1))
        return list(range(-self.half, self.half))

    def _ensure_dir(self, dir_path: str) -> None:
        if dir_path in self._dir_cache:
            return
        files = [
            os.path.join(dir_path, fname)
            for fname in sorted(os.listdir(dir_path))
            if os.path.isfile(os.path.join(dir_path, fname))
        ]
        self._dir_cache[dir_path] = files
        self._dir_index[dir_path] = {os.path.basename(p): i for i, p in enumerate(files)}

    def get_window(self, center_path: str) -> List[str]:
        if self.temporal_T == 1:
            return [center_path]

        dir_path = os.path.dirname(center_path)
        self._ensure_dir(dir_path)

        files = self._dir_cache.get(dir_path, [])
        idx_map = self._dir_index.get(dir_path, {})
        center_name = os.path.basename(center_path)
        center_idx = idx_map.get(center_name, None)
        if center_idx is None:
            return [center_path for _ in range(self.temporal_T)]

        # Clamp outside-the-range offsets to the closest available frame so the
        # sequence stays monotonic in time even near the boundaries.
        window: List[str] = []
        for offset in self._offsets():
            stride_offset = offset * self.temporal_stride
            candidate_idx = max(0, min(center_idx + stride_offset, len(files) - 1))
            candidate = files[candidate_idx] if os.path.exists(files[candidate_idx]) else center_path
            window.append(candidate)

        return window
