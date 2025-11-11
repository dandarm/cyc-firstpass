import os, argparse, yaml, csv, cv2, json
import numpy as np
import torch
from cyclone_locator.models.simplebaseline import SimpleBaseline
from cyclone_locator.transforms.letterbox import letterbox_image, inverse_map_xy
from cyclone_locator.utils.metrics import peak_and_width
from cyclone_locator.utils.geometry import crop_square

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--frames_glob", default=None, help="se vuoi inferenza libera, altrimenti usa manifest_test dal config")
    return ap.parse_args()

def load_images_list(cfg, frames_glob):
    import glob, pandas as pd
    if frames_glob:
        return sorted(glob.glob(frames_glob))
    else:
        df = pd.read_csv(cfg["data"]["manifest_test"])
        return df["image_path"].tolist()

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    os.makedirs(cfg["infer"]["out_dir"], exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = SimpleBaseline(backbone=cfg["train"]["backbone"])
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.cuda().eval()

    frames = load_images_list(cfg, args.frames_glob)

    csv_path = os.path.join(cfg["infer"]["out_dir"], "preds.csv")
    roi_dir = os.path.join(cfg["infer"]["out_dir"], "roi"); os.makedirs(roi_dir, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path","presence_prob","x_g","y_g","x_orig","y_orig","r_crop_px","roi_path"])
        for p in frames:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE) or cv2.imread(p)
            lb, meta = letterbox_image(img, cfg["train"]["image_size"])

            x = lb.astype(np.float32)/255.0
            if x.ndim == 2: x = x[...,None]
            x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).cuda()

            with torch.no_grad():
                hm_p, pr_logit = model(x)

            pr = sigmoid(pr_logit.squeeze(0).squeeze(0).cpu().numpy())
            H = hm_p.squeeze(0).squeeze(0).cpu().numpy()

            if pr >= cfg["infer"]["presence_threshold"]:
                yh, xh, M, width = peak_and_width(H, rel=0.6)
                # coord nel 512: moltiplica per stride (4)
                x_g = (xh * cfg["train"]["heatmap_stride"])
                y_g = (yh * cfg["train"]["heatmap_stride"])
                x_orig, y_orig = inverse_map_xy(x_g, y_g, meta)

                # raggio crop
                r = max(cfg["infer"]["roi_base_radius_px"],
                        int(round(cfg["infer"]["roi_sigma_multiplier"] * width * cfg["train"]["heatmap_stride"])))

                roi = crop_square(img, (x_orig, y_orig), r)
                roi_path = os.path.join(roi_dir, os.path.basename(p))
                cv2.imwrite(roi_path, roi)
            else:
                x_g = y_g = x_orig = y_orig = ""
                r = ""
                roi_path = ""

            w.writerow([p, f"{pr:.4f}", x_g, y_g, x_orig, y_orig, r, roi_path])

    print("Preds saved:", csv_path)
    print("ROI dir:", roi_dir)

    # Hook per integrazione HR: qui puoi chiamare il tuo script VideoMAE
    # emit_roi_for_hr(roi_dir, ...)

if __name__ == "__main__":
    main()
