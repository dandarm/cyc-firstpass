import cv2
import numpy as np

def overlay_heatmap(image, heatmap, alpha=0.5):
    H = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    H_norm = (H - H.min()) / (H.max() - H.min() + 1e-6)
    colored = cv2.applyColorMap((H_norm*255).astype(np.uint8), cv2.COLORMAP_JET)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(image, 1-alpha, colored, alpha, 0)
    return out
