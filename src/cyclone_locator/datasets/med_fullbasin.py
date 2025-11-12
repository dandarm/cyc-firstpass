import os, cv2, torch, numpy as np, pandas as pd
from torch.utils.data import Dataset
from cyclone_locator.transforms.letterbox import letterbox_image, forward_map_xy

def make_gaussian_heatmap(H, W, cx, cy, sigma):
    """Heatmap gaussiana centrata in (cx,cy) su mappa HxW (float32)."""
    yy, xx = np.mgrid[0:H, 0:W]
    g = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2*sigma*sigma))
    return g.astype(np.float32)

class MedFullBasinDataset(Dataset):
    """
    Legge un CSV manifest con: image_path (ORIG), presence (0/1), cx, cy (in pixel originali).
    Se use_pre_letterboxed=True, fa join con letterbox_meta.csv per ottenere resized_path e meta.
    Crea target heatmap alla risoluzione (S,S) con S=image_size//stride.
    """
    def __init__(self, csv_path, image_size=512, heatmap_stride=4,
                 heatmap_sigma_px=8, use_aug=False,
                 use_pre_letterboxed=True, letterbox_meta_csv=None, letterbox_size_assert=None):
        self.df = pd.read_csv(csv_path)
        self.image_size = int(image_size)
        self.stride = int(heatmap_stride)
        self.Ho = self.image_size // self.stride
        self.Wo = self.image_size // self.stride
        self.sigma = float(heatmap_sigma_px)
        self.use_aug = bool(use_aug)
        self.use_pre_lb = bool(use_pre_letterboxed)

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

        # normalizza path originali a absolute per la join
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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        orig_abs = row["image_path_abs"]

        # carica immagine già letterbox + meta
        if self.use_pre_lb:
            lb, meta = self._load_resized_and_meta(orig_abs)
            if meta["out_size"] != self.image_size:
                raise ValueError(f"image_size mismatch: dataset={self.image_size} vs meta={meta['out_size']}")
        else:
            # fallback (sconsigliato): se proprio vuoi fare on-the-fly
            raise RuntimeError("use_pre_letterboxed=False non supportato in questa configurazione")

        presence = int(row["presence"])
        if presence == 1:
            cx, cy = float(row["cx"]), float(row["cy"])
            xg, yg = self._forward_map_xy(cx, cy, meta)
        else:
            xg, yg = -1.0, -1.0

        # augment minimi (opzionali)
        if self.use_aug:
            if np.random.rand() < 0.5:
                lb = np.fliplr(lb).copy()
                if presence == 1:
                    xg = self.image_size - 1 - xg

        # to tensor [0,1], (C,H,W)
        if lb.ndim == 2:
            lb = lb[..., None]
        lb = lb.astype(np.float32) / 255.0
        img_t = torch.from_numpy(lb).permute(2,0,1)

        # target heatmap a risoluzione ridotta
        if presence == 1:
            cx_hm = xg / self.stride
            cy_hm = yg / self.stride
            hm = make_gaussian_heatmap(self.Ho, self.Wo, cx_hm, cy_hm, self.sigma / self.stride)
        else:
            hm = np.zeros((self.Ho, self.Wo), dtype=np.float32)

        sample = {
            "image": img_t,                        # (C,H,W) float32
            "heatmap": torch.from_numpy(hm)[None], # (1,Ho,Wo)
            "presence": torch.tensor([presence], dtype=torch.float32),
            # meta serve solo in inferenza; in training teniamo lo stretto necessario
            "meta_scale": meta["scale"],
            "meta_pad_x": meta["pad_x"],
            "meta_pad_y": meta["pad_y"],
            "orig_w": meta["orig_w"],
            "orig_h": meta["orig_h"],
            "image_path": row["image_path"]
        }
        return sample