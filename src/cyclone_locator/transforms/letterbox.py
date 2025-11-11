import cv2
import numpy as np

def letterbox_image(img, out_size):
    """Ridimensiona mantenendo AR + padding per ottenere out_size x out_size.
    Ritorna: img_lb, meta con (orig_w,h), (w',h'), scale, pad_x, pad_y."""
    H, W = img.shape[:2]
    s = out_size / max(W, H)
    w_new, h_new = int(round(W * s)), int(round(H * s))
    img_resized = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
    pad_x = (out_size - w_new) // 2
    pad_y = (out_size - h_new) // 2
    if img.ndim == 2:
        img_lb = np.full((out_size, out_size), 0, dtype=img_resized.dtype)
        img_lb[pad_y:pad_y+h_new, pad_x:pad_x+w_new] = img_resized
    else:
        img_lb = np.zeros((out_size, out_size, img.shape[2]), dtype=img_resized.dtype)
        img_lb[pad_y:pad_y+h_new, pad_x:pad_x+w_new, :] = img_resized
    meta = dict(orig_w=W, orig_h=H, w_new=w_new, h_new=h_new, scale=s, pad_x=pad_x, pad_y=pad_y)
    return img_lb, meta

def forward_map_xy(x_orig, y_orig, meta):
    """(x_orig,y_orig) → (x_g,y_g) nello spazio letterbox out_size."""
    xg = meta["scale"] * x_orig + meta["pad_x"]
    yg = meta["scale"] * y_orig + meta["pad_y"]
    return xg, yg

def inverse_map_xy(xg, yg, meta):
    """(x_g,y_g) → (x_orig,y_orig) nelle dimensioni native dell'immagine."""
    x_orig = (xg - meta["pad_x"]) / meta["scale"]
    y_orig = (yg - meta["pad_y"]) / meta["scale"]
    return x_orig, y_orig
