import os, cv2, torch, numpy as np, pandas as pd
from torch.utils.data import Dataset

from cyclone_locator.datasets.temporal_utils import TemporalWindowSelector

def make_gaussian_heatmap(H, W, cx, cy, sigma):
    """Heatmap gaussiana centrata in (cx,cy) su mappa HxW (float32)."""
    yy, xx = np.mgrid[0:H, 0:W]
    g = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2*sigma*sigma))
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
                 temporal_T=1, temporal_stride=1):
        self.df = pd.read_csv(csv_path)
        self.image_size = int(image_size)
        self.stride = int(heatmap_stride)
        self.Ho = self.image_size // self.stride
        self.Wo = self.image_size // self.stride
        self.sigma = float(heatmap_sigma_px)
        self.use_aug = bool(use_aug)
        self.temporal_selector = TemporalWindowSelector(temporal_T, temporal_stride)

        self.df["image_path"] = self.df["image_path"].astype(str)

        cols = set(self.df.columns)
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
        self.df["image_path_abs"] = self.df["image_path"].apply(lambda p: os.path.abspath(p))

    def __len__(self):
        return len(self.df)

    def _load_resized_and_meta(self, orig_abs):
        r = self.meta_map.get(orig_abs, None)
        if r is None:
            raise KeyError(f"no letterbox meta for {orig_abs}")
        img = cv2.imread(r["resized_path"], cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(r["resized_path"])
        meta = dict(scale=float(r["scale"]),
                    pad_x=float(r["pad_x"]),
                    pad_y=float(r["pad_y"]),
                    out_size=int(r["out_size"]),
                    orig_w=int(r["orig_w"]),
                    orig_h=int(r["orig_h"]))
        return img, meta

    @staticmethod
    def _forward_map_xy(x_orig, y_orig, meta):
        xg = meta["scale"] * x_orig + meta["pad_x"]
        yg = meta["scale"] * y_orig + meta["pad_y"]
        return xg, yg

    def _load_letterboxed(self, abs_path):
        if self.letterboxed_manifest:
            lb = cv2.imread(abs_path, cv2.IMREAD_UNCHANGED)
            if lb is None:
                raise FileNotFoundError(abs_path)
            meta = dict(scale=1.0, pad_x=0.0, pad_y=0.0,
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
        row = self.df.iloc[idx]
        orig_abs = row["image_path_abs"]

        center_img, meta = self._load_letterboxed(orig_abs)
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

        presence = int(row["presence"])
        if presence == 1:
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
            if presence == 1:
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
        if presence == 1:
            cx_hm = xg / self.stride
            cy_hm = yg / self.stride
            hm = torch.from_numpy(
                make_gaussian_heatmap(self.Ho, self.Wo, cx_hm, cy_hm, self.sigma / self.stride)
            )
        else:
            noise_level = 0.01
            hm = torch.rand((self.Ho, self.Wo), dtype=torch.float32) * noise_level
            hm = torch.clamp(hm, 0.0, 0.02)

        sample = {
            "image": img_t,                        # (C,H,W) float32 (temporal early fusion)
            "video": video_t,                      # (C,T,H,W) float32 (explicit temporal dim)
            "heatmap": hm.unsqueeze(0),            # (1,Ho,Wo)
            "presence": torch.tensor([presence], dtype=torch.float32),
            # meta serve solo in inferenza; in training teniamo lo stretto necessario
            "meta_scale": meta["scale"],
            "meta_pad_x": meta["pad_x"],
            "meta_pad_y": meta["pad_y"],
            "orig_w": meta["orig_w"],
            "orig_h": meta["orig_h"],
            "image_path": row["image_path"],
            "image_path_abs": orig_abs
        }

        return sample
