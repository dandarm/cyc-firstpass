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
    Legge un CSV con: image_path, presence (0/1), cx, cy (in pixel originali, se presence=1).
    Applica letterbox a 512x512, crea target heatmap a risoluzione (S,S) dove S=512//stride.
    """
    def __init__(self, csv_path, image_size=512, heatmap_stride=4,
                 heatmap_sigma_px=8, use_aug=False):
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size
        self.stride = heatmap_stride
        self.Ho = image_size // heatmap_stride
        self.Wo = image_size // heatmap_stride
        self.sigma = heatmap_sigma_px
        self.use_aug = use_aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE) or \
              cv2.imread(row["image_path"])
        if img is None:
            raise FileNotFoundError(row["image_path"])
        lb, meta = letterbox_image(img, self.image_size)

        presence = int(row["presence"])
        if presence == 1:
            cx, cy = float(row["cx"]), float(row["cy"])
            xg, yg = forward_map_xy(cx, cy, meta)
        else:
            xg, yg = -1.0, -1.0

        # augment minimi (opzionali)
        if self.use_aug:
            # esempio: nessuna distorsione, solo flip orizz. con probabilità 0.5
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
            "meta": meta,
            "image_path": row["image_path"],
            "xg": xg, "yg": yg
        }
        return sample
