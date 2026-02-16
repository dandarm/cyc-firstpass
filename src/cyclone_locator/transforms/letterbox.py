import cv2
import numpy as np

def resize_image(img, out_size: int, mode: str = "letterbox"):
    """Resize immagine a ``out_size x out_size``.

    mode:
      - ``letterbox``: mantiene aspect ratio + padding
      - ``stretch``: resize diretto senza preservare aspect ratio (distorsione), senza padding

    Ritorna: img_out, meta con (orig_w,h), (w_new,h_new), pad_x/pad_y e scale.
    Per compatibilità, meta include sempre ``scale``; se ``mode=stretch`` include anche
    ``scale_x`` e ``scale_y`` (non uniformi).
    """
    out_size = int(out_size)
    H, W = img.shape[:2]
    if mode not in {"letterbox", "stretch"}:
        raise ValueError("mode must be 'letterbox' or 'stretch'")

    if mode == "stretch":
        w_new, h_new = out_size, out_size
        img_resized = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
        pad_x, pad_y = 0, 0
        scale_x = out_size / float(W)
        scale_y = out_size / float(H)
        meta = dict(
            mode=mode,
            orig_w=W,
            orig_h=H,
            w_new=w_new,
            h_new=h_new,
            scale=float(scale_x),  # legacy field: do not use for Y when scale_y differs
            scale_x=float(scale_x),
            scale_y=float(scale_y),
            pad_x=float(pad_x),
            pad_y=float(pad_y),
        )
        return img_resized, meta

    # letterbox
    s = out_size / max(W, H)
    w_new, h_new = int(round(W * s)), int(round(H * s))
    img_resized = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
    pad_x = (out_size - w_new) // 2
    pad_y = (out_size - h_new) // 2
    if img.ndim == 2:
        img_out = np.full((out_size, out_size), 0, dtype=img_resized.dtype)
        img_out[pad_y : pad_y + h_new, pad_x : pad_x + w_new] = img_resized
    else:
        img_out = np.zeros((out_size, out_size, img.shape[2]), dtype=img_resized.dtype)
        img_out[pad_y : pad_y + h_new, pad_x : pad_x + w_new, :] = img_resized
    meta = dict(
        mode=mode,
        orig_w=W,
        orig_h=H,
        w_new=w_new,
        h_new=h_new,
        scale=float(s),
        scale_x=float(s),
        scale_y=float(s),
        pad_x=float(pad_x),
        pad_y=float(pad_y),
    )
    return img_out, meta


def letterbox_image(img, out_size):
    """Compat: vecchia API che esegue ``mode=letterbox``."""
    return resize_image(img, out_size, mode="letterbox")

def forward_map_xy(x_orig, y_orig, meta):
    """(x_orig,y_orig) → (x_g,y_g) nello spazio letterbox out_size."""
    sx = float(meta.get("scale_x", meta["scale"]))
    sy = float(meta.get("scale_y", meta["scale"]))
    xg = sx * x_orig + float(meta.get("pad_x", 0.0))
    yg = sy * y_orig + float(meta.get("pad_y", 0.0))
    return xg, yg

def inverse_map_xy(xg, yg, meta):
    """(x_g,y_g) → (x_orig,y_orig) nelle dimensioni native dell'immagine."""
    sx = float(meta.get("scale_x", meta["scale"]))
    sy = float(meta.get("scale_y", meta["scale"]))
    x_orig = (xg - float(meta.get("pad_x", 0.0))) / sx
    y_orig = (yg - float(meta.get("pad_y", 0.0))) / sy
    return x_orig, y_orig
