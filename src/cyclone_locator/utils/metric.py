import numpy as np

def mae_km(pred_xy, true_xy, pix2km):
    """Media dell'errore assoluto in km (lista di coppie)."""
    errs = []
    for p, t in zip(pred_xy, true_xy):
        if p is None or t is None:
            continue
        dx = (p[0] - t[0]) * pix2km
        dy = (p[1] - t[1]) * pix2km
        errs.append((dx**2 + dy**2)**0.5)
    return float(np.mean(errs)) if errs else np.nan

def pct_within_Rkm(pred_xy, true_xy, pix2km, R):
    count, tot = 0, 0
    for p, t in zip(pred_xy, true_xy):
        if p is None or t is None:
            continue
        dx = (p[0] - t[0]) * pix2km
        dy = (p[1] - t[1]) * pix2km
        d = (dx**2 + dy**2)**0.5
        tot += 1
        if d <= R:
            count += 1
    return (100.0 * count / tot) if tot else np.nan

def peak_and_width(H, rel=0.6):
    """Ritorna (y,x, M, width_px) dal picco della heatmap H (2D)."""
    import numpy as np
    idx = np.argmax(H)
    Hh, Hw = H.shape
    y = idx // Hw
    x = idx %  Hw
    M = H[y, x]
    W = H[max(0,y-5):min(Hh,y+6), max(0,x-5):min(Hw,x+6)]
    width = (W > rel*M).sum()**0.5
    return y, x, float(M), float(width)
