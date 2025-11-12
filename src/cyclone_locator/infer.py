# src/cyclone_locator/infer.py
import os, argparse, yaml, csv, cv2
import numpy as np
import pandas as pd
import torch
from cyclone_locator.models.simplebaseline import SimpleBaseline
from cyclone_locator.utils.metrics import peak_and_width
from cyclone_locator.utils.geometry import crop_square

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--frames_glob", default=None,
                    help="(opzionale) se specifichi, verrà filtrato il meta solo per questi file")
    return ap.parse_args()

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    out_dir = cfg["infer"]["out_dir"] if args.out_dir is None else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # carica meta letterbox
    meta_df = pd.read_csv(cfg["data"]["letterbox_meta_csv"])
    meta_df["orig_path_abs"] = meta_df["orig_path"].apply(lambda p: os.path.abspath(p))
    meta_map = {r["orig_path_abs"]: r for _, r in meta_df.iterrows()}

    # carica lista immagini
    if args.frames_glob:
        import glob
        orig_list = [os.path.abspath(p) for p in sorted(glob.glob(args.frames_glob))]
        # filtra meta per i soli file selezionati
        meta_rows = [meta_map[p] for p in orig_list if p in meta_map]
    else:
        meta_rows = list(meta_map.values())

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = SimpleBaseline(backbone=cfg["train"]["backbone"])
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.cuda().eval()

    csv_path = os.path.join(out_dir, "preds.csv")
    roi_dir = os.path.join(out_dir, "roi"); os.makedirs(roi_dir, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["orig_path","resized_path","presence_prob","x_g","y_g","x_orig","y_orig","r_crop_px","roi_path"])
        for r in meta_rows:
            op = r["orig_path_abs"]
            rp = r["resized_path"]
            img = cv2.imread(rp, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[WARN] cannot read resized: {rp}")
                continue

            x = img.astype(np.float32) / 255.0
            if x.ndim == 2: x = x[...,None]
            x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).cuda()

            with torch.no_grad():
                hm_p, pr_logit = model(x)
            pr = sigmoid(pr_logit.squeeze(0).squeeze(0).cpu().numpy())
            H = hm_p.squeeze(0).squeeze(0).cpu().numpy()

            if pr >= cfg["infer"]["presence_threshold"]:
                yh, xh, M, width = peak_and_width(H, rel=0.6)
                x_g = (xh * cfg["train"]["heatmap_stride"])
                y_g = (yh * cfg["train"]["heatmap_stride"])

                # inversa verso i pixel originali
                scale = float(r["scale"]); pad_x = float(r["pad_x"]); pad_y = float(r["pad_y"])
                x_orig = (x_g - pad_x) / scale
                y_orig = (y_g - pad_y) / scale

                # carica originale per ritaglio ROI
                orig_img = cv2.imread(op, cv2.IMREAD_UNCHANGED)
                if orig_img is None:
                    print(f"[WARN] cannot read original: {op}")
                    roi_path = ""; r_crop = ""
                else:
                    r_crop = max(cfg["infer"]["roi_base_radius_px"],
                                 int(round(cfg["infer"]["roi_sigma_multiplier"] * width * cfg["train"]["heatmap_stride"])))
                    roi = crop_square(orig_img, (x_orig, y_orig), r_crop)
                    roi_path = os.path.join(roi_dir, os.path.basename(op))
                    cv2.imwrite(roi_path, roi)
            else:
                x_g = y_g = x_orig = y_orig = ""
                r_crop = ""; roi_path = ""

            w.writerow([op, rp, f"{pr:.4f}", x_g, y_g, x_orig, y_orig, r_crop, roi_path])

    print("Preds saved:", csv_path)
    print("ROI dir:", roi_dir)

if __name__ == "__main__":
    main()
