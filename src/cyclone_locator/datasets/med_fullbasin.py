import os, warnings, cv2, torch, numpy as np, pandas as pd
from torch.utils.data import Dataset

from cyclone_locator.datasets.temporal_utils import TemporalWindowSelector

def make_gaussian_heatmap_from_grid(xx, yy, cx, cy, sigma):
    """Heatmap gaussiana centrata in (cx,cy) su griglie (xx,yy) (float32)."""
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    return g.astype(np.float32)

class MedFullBasinDataset(Dataset):
    """
    Legge un CSV manifest con: image_path, presence (0/1) e coordinate target.
    Supporta sia manifest con path originali + meta letterbox, sia manifest già letterbox
    con colonne x_pix_resized/y_pix_resized.
    """
    def __init__(self, csv_path, image_size=512, heatmap_stride=4,
                 heatmap_sigma_px=8, use_aug=False,
                 use_pre_letterboxed=True, letterbox_meta_csv=None, letterbox_size_assert=None,
                 temporal_T=1, temporal_stride=1,
                 manifest_stride: int = 1):
        full_df = pd.read_csv(csv_path)
        self.image_size = int(image_size)
        self.stride = int(heatmap_stride)
        self.Ho = self.image_size // self.stride
        self.Wo = self.image_size // self.stride
        self.sigma = float(heatmap_sigma_px)
        self.use_aug = bool(use_aug)
        self.temporal_selector = TemporalWindowSelector(temporal_T, temporal_stride)
        # In caso di file mancanti/corrotti, possiamo fare retry su un altro indice
        # per evitare che il DataLoader termini con eccezione.
        self.max_missing_retries = 25
        self._missing_warned = 0

        yy, xx = np.mgrid[0 : self.Ho, 0 : self.Wo]
        self._hm_xx = xx.astype(np.float32)
        self._hm_yy = yy.astype(np.float32)
        self._hm_xx_t = torch.from_numpy(self._hm_xx)
        self._hm_yy_t = torch.from_numpy(self._hm_yy)

        full_df["image_path"] = full_df["image_path"].astype(str)

        cols = set(full_df.columns)
        self._has_resized_keypoints = {"x_pix_resized", "y_pix_resized"}.issubset(cols)
        self._has_orig_keypoints = {"cx", "cy"}.issubset(cols)
        if not self._has_resized_keypoints and not self._has_orig_keypoints:
            raise ValueError("manifest must include either cx/cy or x_pix_resized/y_pix_resized columns")

        # Se il manifest contiene path già letterbox, non serve meta CSV
        self.letterboxed_manifest = self._has_resized_keypoints
        self.use_pre_lb = bool(use_pre_letterboxed) and not self.letterboxed_manifest

        self.meta_map = None
        if self.use_pre_lb:
            if not letterbox_meta_csv or not os.path.exists(letterbox_meta_csv):
                raise FileNotFoundError(f"letterbox_meta_csv not found: {letterbox_meta_csv}")
            meta_df = pd.read_csv(letterbox_meta_csv)
            if letterbox_size_assert is not None:
                uniq = set(meta_df["out_size"].unique().tolist())
                if len(uniq) != 1 or (letterbox_size_assert not in uniq):
                    raise ValueError(f"letterbox out_size {uniq} != expected {letterbox_size_assert}")
            # indicizza per percorso originale assoluto
            meta_df["orig_path_abs"] = meta_df["orig_path"].apply(lambda p: os.path.abspath(p))
            self.meta_map = {r["orig_path_abs"]: r for _, r in meta_df.iterrows()}

        # normalizza path a absolute per la join
        full_df["image_path_abs"] = full_df["image_path"].apply(lambda p: os.path.abspath(p))

        manifest_stride = int(manifest_stride) if manifest_stride is not None else 1
        if manifest_stride < 1:
            manifest_stride = 1
        if manifest_stride > 1:
            self.df = full_df.iloc[::manifest_stride].reset_index(drop=True)
        else:
            self.df = full_df

        # mappe ausiliarie per presenza/coordinate per frame
        self._frame_info = {}
        for row in full_df.itertuples(index=False):
            abs_path = getattr(row, "image_path_abs")
            info = {"presence": int(getattr(row, "presence"))}
            if self._has_resized_keypoints:
                info["x_pix_resized"] = getattr(row, "x_pix_resized")
                info["y_pix_resized"] = getattr(row, "y_pix_resized")
            if self._has_orig_keypoints:
                info["cx"] = getattr(row, "cx")
                info["cy"] = getattr(row, "cy")
            self._frame_info[abs_path] = info

        self._presence_map = {p: info["presence"] for p, info in self._frame_info.items()}

    def _meta_for_path_abs(self, abs_path: str) -> dict:
        if self.letterboxed_manifest:
            return dict(scale=1.0, scale_x=1.0, scale_y=1.0, pad_x=0.0, pad_y=0.0,
                        out_size=self.image_size, orig_w=self.image_size, orig_h=self.image_size)
        if self.use_pre_lb:
            r = self.meta_map.get(abs_path, None)
            if r is None:
                raise KeyError(f"no letterbox meta for {abs_path}")
            scale = float(r["scale"])
            scale_x = float(r["scale_x"]) if "scale_x" in r else scale
            scale_y = float(r["scale_y"]) if "scale_y" in r else scale
            return dict(scale=scale,
                        scale_x=scale_x,
                        scale_y=scale_y,
                        pad_x=float(r["pad_x"]),
                        pad_y=float(r["pad_y"]),
                        out_size=int(r["out_size"]),
                        orig_w=int(r["orig_w"]),
                        orig_h=int(r["orig_h"]))
        raise RuntimeError("use_pre_letterboxed=False non supportato in questa configurazione")

    def _presence_probability(self, window_paths, default_presence):
        """Compute a probabilistic presence based on the full temporal span.

        Instead of counting only the selected frames, we collect all manifest
        rows between the earliest and latest frame of the window. This captures
        the actual temporal coverage implied by the stride, yielding smoother
        probabilities near label transitions.
        """

        if not window_paths:
            return float(default_presence)

        dir_path = os.path.dirname(window_paths[0])
        self.temporal_selector._ensure_dir(dir_path)
        files = self.temporal_selector._dir_cache.get(dir_path, [])
        idx_map = self.temporal_selector._dir_index.get(dir_path, {})

        indices = [idx_map.get(os.path.basename(p)) for p in window_paths if os.path.basename(p) in idx_map]
        if not indices:
            return float(default_presence)

        start, end = min(indices), max(indices)
        values = []
        for i in range(start, end + 1):
            path = files[i]
            abs_path = os.path.abspath(path)
            values.append(self._presence_map.get(abs_path, default_presence))

        if not values:
            return float(default_presence)
        return float(np.mean(values))

    def _span_abs_paths_from_window(self, window_paths):
        if not window_paths:
            return []
        dir_path = os.path.dirname(window_paths[0])
        self.temporal_selector._ensure_dir(dir_path)
        files = self.temporal_selector._dir_cache.get(dir_path, [])
        idx_map = self.temporal_selector._dir_index.get(dir_path, {})
        indices = [idx_map.get(os.path.basename(p)) for p in window_paths if os.path.basename(p) in idx_map]
        if not indices:
            return []
        start, end = min(indices), max(indices)
        return [os.path.abspath(files[i]) for i in range(start, end + 1)]

    def _keypoint_lb_from_abs(self, abs_path: str):
        info = self._frame_info.get(abs_path)
        if not info:
            return None
        if int(info.get("presence", 0)) != 1:
            return None

        if self.letterboxed_manifest and self._has_resized_keypoints:
            xg = info.get("x_pix_resized")
            yg = info.get("y_pix_resized")
            if xg is None or yg is None or pd.isna(xg) or pd.isna(yg):
                return None
            return float(xg), float(yg)

        if self._has_orig_keypoints:
            cx = info.get("cx")
            cy = info.get("cy")
            if cx is None or cy is None or pd.isna(cx) or pd.isna(cy):
                return None
            meta = self._meta_for_path_abs(abs_path)
            xg, yg = self._forward_map_xy(float(cx), float(cy), meta)
            return float(xg), float(yg)

        return None

    def _soft_target_heatmap(self, window_paths, presence_prob: float, fallback_xy_lb=None):
        """Build a soft heatmap target over the full temporal span.

        The returned heatmap has max value ~= presence_prob, so the peak is
        numerically congruent with the (soft) presence label.
        """
        p = float(np.clip(presence_prob, 0.0, 1.0))
        if p <= 0.0:
            return None

        span_abs_paths = self._span_abs_paths_from_window(window_paths)
        coords = []
        for abs_path in span_abs_paths:
            xy = self._keypoint_lb_from_abs(abs_path)
            if xy is not None:
                coords.append(xy)

        if not coords and fallback_xy_lb is not None:
            coords = [fallback_xy_lb]

        if not coords:
            return torch.zeros((self.Ho, self.Wo), dtype=torch.float32)

        acc = np.zeros((self.Ho, self.Wo), dtype=np.float32)
        sigma_hm = self.sigma / self.stride
        for xg, yg in coords:
            cx_hm = xg / self.stride
            cy_hm = yg / self.stride
            acc += make_gaussian_heatmap_from_grid(self._hm_xx, self._hm_yy, cx_hm, cy_hm, sigma_hm)

        mx = float(acc.max())
        if mx > 0:
            acc = (acc / mx) * p
        else:
            acc[:] = 0.0
        return torch.from_numpy(acc)

    def __len__(self):
        return len(self.df)

    def _load_resized_and_meta(self, orig_abs):
        r = self.meta_map.get(orig_abs, None)
        if r is None:
            raise KeyError(f"no letterbox meta for {orig_abs}")
        resized_path = r["resized_path"]
        img = cv2.imread(resized_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"resized not found/readable: {resized_path} (orig: {orig_abs})")
        scale = float(r["scale"])
        scale_x = float(r["scale_x"]) if "scale_x" in r else scale
        scale_y = float(r["scale_y"]) if "scale_y" in r else scale
        meta = dict(scale=scale,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    pad_x=float(r["pad_x"]),
                    pad_y=float(r["pad_y"]),
                    out_size=int(r["out_size"]),
                    orig_w=int(r["orig_w"]),
                    orig_h=int(r["orig_h"]))
        return img, meta

    @staticmethod
    def _forward_map_xy(x_orig, y_orig, meta):
        sx = float(meta.get("scale_x", meta["scale"]))
        sy = float(meta.get("scale_y", meta["scale"]))
        xg = sx * x_orig + meta["pad_x"]
        yg = sy * y_orig + meta["pad_y"]
        return xg, yg

    def _load_letterboxed(self, abs_path):
        if self.letterboxed_manifest:
            lb = cv2.imread(abs_path, cv2.IMREAD_UNCHANGED)
            if lb is None:
                raise FileNotFoundError(abs_path)
            meta = dict(scale=1.0, scale_x=1.0, scale_y=1.0, pad_x=0.0, pad_y=0.0,
                        out_size=self.image_size, orig_w=self.image_size, orig_h=self.image_size)
        elif self.use_pre_lb:
            lb, meta = self._load_resized_and_meta(abs_path)
        else:
            raise RuntimeError("use_pre_letterboxed=False non supportato in questa configurazione")
        if lb is None:
            raise FileNotFoundError(abs_path)
        if lb.shape[0] != self.image_size or lb.shape[1] != self.image_size:
            raise ValueError(
                f"letterboxed image size {lb.shape[:2]} != expected square {self.image_size}"
            )
        if meta["out_size"] != self.image_size:
            raise ValueError(f"image_size mismatch: dataset={self.image_size} vs meta={meta['out_size']}")
        return lb, meta

    def _normalize_frame(self, frame_np):
        if frame_np.ndim == 2:
            frame_np = frame_np[..., None]
        frame_np = frame_np.astype(np.float32)
        return frame_np / 255.0

    def __getitem__(self, idx):
        # Se l'immagine centrale manca/cv2 non riesce a leggerla, non abortire:
        # prova un altro indice e continua.
        last_err = None
        row = None
        orig_abs = None
        center_img = None
        meta = None
        for k in range(max(1, int(self.max_missing_retries))):
            row = self.df.iloc[idx]
            orig_abs = row["image_path_abs"]
            try:
                center_img, meta = self._load_letterboxed(orig_abs)
                break
            except (FileNotFoundError, KeyError, ValueError) as e:
                last_err = e
                if self._missing_warned < 10:
                    warnings.warn(f"[MedFullBasinDataset] skip missing sample idx={idx} path={orig_abs}: {e}")
                    self._missing_warned += 1
                idx = (int(idx) + 1) % len(self.df)
        else:
            raise RuntimeError(
                f"Too many missing/corrupt samples (last idx={idx}, path={orig_abs}): {last_err}"
            )

        window_paths = self.temporal_selector.get_window(orig_abs)

        frames = []
        for path in window_paths:
            if path == orig_abs:
                img_np = center_img
            else:
                try:
                    img_np, _ = self._load_letterboxed(path)
                except Exception:
                    img_np = center_img
            if img_np.shape != center_img.shape:
                img_np = center_img
            frames.append(img_np)

        #print(np.mean((frames[0] - frames[1])**2).item(),
        #      np.mean((frames[1] - frames[2])**2).item())

        presence_center = int(row["presence"])
        presence_prob = self._presence_probability(window_paths, presence_center)

        if presence_center == 1:
            if self.letterboxed_manifest and self._has_resized_keypoints and \
               not pd.isna(row["x_pix_resized"]) and not pd.isna(row["y_pix_resized"]):
                xg = float(row["x_pix_resized"])
                yg = float(row["y_pix_resized"])
            elif self._has_orig_keypoints and not pd.isna(row["cx"]) and not pd.isna(row["cy"]):
                cx, cy = float(row["cx"]), float(row["cy"])
                xg, yg = self._forward_map_xy(cx, cy, meta)
            else:
                raise ValueError("Positive sample without keypoint coordinates")
        else:
            xg, yg = -1.0, -1.0

        # augment minimi (opzionali)
        if self.use_aug and np.random.rand() < 0.5:
            frames = [np.fliplr(f).copy() for f in frames]
            if presence_center == 1:
                xg = self.image_size - 1 - xg

        frames = [self._normalize_frame(f) for f in frames]
        ch = frames[0].shape[2]
        fused = np.concatenate(frames, axis=2)
        if fused.shape[2] != ch * len(frames):
            raise ValueError("Unexpected channel mismatch in temporal fusion")
        img_t = torch.from_numpy(fused).permute(2,0,1)
        video_t = torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in frames], dim=0)
        video_t = video_t.permute(1, 0, 2, 3)  # (C,T,H,W)

        # target heatmap a risoluzione ridotta
        if float(presence_prob) > 0.0:
            fallback = (float(xg), float(yg)) if presence_center == 1 else None
            hm = self._soft_target_heatmap(window_paths, presence_prob, fallback_xy_lb=fallback)
        else:
            hm = torch.zeros((self.Ho, self.Wo), dtype=torch.float32)

        hm_sum = float(hm.sum().item())
        if hm_sum > 0.0:
            x_hm = float((hm * self._hm_xx_t).sum().item() / hm_sum)
            y_hm = float((hm * self._hm_yy_t).sum().item() / hm_sum)
            target_xy_valid = 1.0
        else:
            x_hm = -1.0
            y_hm = -1.0
            target_xy_valid = 0.0

        sample = {
            "image": img_t,                        # (C,H,W) float32 (temporal early fusion)
            "video": video_t,                      # (C,T,H,W) float32 (explicit temporal dim)
            "heatmap": hm.unsqueeze(0),            # (1,Ho,Wo)
            "presence": torch.tensor([presence_prob], dtype=torch.float32),
            "target_xy_hm": torch.tensor([x_hm, y_hm], dtype=torch.float32),
            "target_xy_valid": torch.tensor([target_xy_valid], dtype=torch.float32),
            # meta serve solo in inferenza; in training teniamo lo stretto necessario
            "meta_scale": meta["scale"],
            "meta_scale_x": meta.get("scale_x", meta["scale"]),
            "meta_scale_y": meta.get("scale_y", meta["scale"]),
            "meta_pad_x": meta["pad_x"],
            "meta_pad_y": meta["pad_y"],
            "orig_w": meta["orig_w"],
            "orig_h": meta["orig_h"],
            "image_path": row["image_path"],
            "image_path_abs": orig_abs
        }

        return sample
