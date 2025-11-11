#!/usr/bin/env python3
import argparse, os, cv2, pandas as pd
from cyclone_locator.transforms.letterbox import letterbox_image, forward_map_xy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--image_size", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.manifest_csv)

    out_rows = []
    for _, row in df.iterrows():
        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE) or cv2.imread(row["image_path"])
        H, W = img.shape[:2]
        lb, meta = letterbox_image(img, args.image_size)
        out_path = os.path.join(args.out_dir, os.path.basename(row["image_path"]))
        cv2.imwrite(out_path, lb)

        cx, cy = row.get("cx", ""), row.get("cy", "")
        if row["presence"] == 1 and cx != "" and cy != "":
            xg, yg = forward_map_xy(float(cx), float(cy), meta)
        else:
            xg, yg = "", ""
        out_rows.append([out_path, row["presence"], xg, yg, W, H, meta["scale"], meta["pad_x"], meta["pad_y"]])

    out_csv = os.path.join(args.out_dir, "manifest_letterboxed.csv")
    pd.DataFrame(out_rows, columns=["image_path","presence","x_g","y_g","W","H","scale","pad_x","pad_y"]).to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

if __name__ == "__main__":
    main()
