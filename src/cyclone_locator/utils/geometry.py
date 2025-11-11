def crop_square(img, center_xy, radius):
    import numpy as np
    x, y = center_xy
    x0, x1 = int(round(x - radius)), int(round(x + radius))
    y0, y1 = int(round(y - radius)), int(round(y + radius))
    H, W = img.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    return img[y0:y1, x0:x1]
