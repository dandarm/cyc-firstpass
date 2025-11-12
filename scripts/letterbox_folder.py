#!/usr/bin/env python3
# scripts/letterbox_folder.py
import argparse, os, sys, csv, pathlib
import cv2

def letterbox_image(img, out_size):
    H, W = img.shape[:2]
    s = out_size / max(W, H)
    w_new, h_new = int(round(W * s)), int(round(H * s))
    img_resized = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
    pad_x = (out_size - w_new) // 2
    pad_y = (out_size - h_new) // 2
    if img.ndim == 2:
        canvas = (img_resized.dtype.type(0))
        img_lb = cv2.copyMakeBorder(img_resized, pad_y, out_size-h_new-pad_y,
                                    pad_x, out_size-w_new-pad_x,
                                    borderType=cv2.BORDER_CONSTANT, value=canvas)
    else:
        img_lb = cv2.copyMakeBorder(img_resized, pad_y, out_size-h_new-pad_y,
                                    pad_x, out_size-w_new-pad_x,
                                    borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    meta = dict(orig_w=W, orig_h=H, out_size=out_size,
                w_new=w_new, h_new=h_new, scale=s, pad_x=pad_x, pad_y=pad_y)
    return img_lb, meta

def iter_images(root, exts):
    root = pathlib.Path(root)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def main():
    ap = argparse.ArgumentParser(description="Pre-compute letterbox for a whole folder")
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--size", type=int, default=512, help="output square size, e.g. 512/448/320/…")
    ap.add_argument("--exts", default=".png,.jpg,.jpeg,.tif,.tiff,.bmp", help="comma-separated")
    ap.add_argument("--preserve_tree", action="store_true",
                    help="replicate the input directory structure under out_dir/resized")
    args = ap.parse_args()

    in_dir  = pathlib.Path(args.in_dir).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    out_img_root = out_dir / "resized"
    out_img_root.mkdir(parents=True, exist_ok=True)

    exts = tuple([e.strip().lower() for e in args.exts.split(",") if e.strip()])

    meta_rows = []
    count = 0
    for ipath in iter_images(in_dir, exts):
        rel = ipath.relative_to(in_dir)
        if args.preserve_tree:
            save_dir = out_img_root / rel.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            opath = save_dir / rel.name
        else:
            opath = out_img_root / rel.name

        img = cv2.imread(str(ipath), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] cannot read: {ipath}", file=sys.stderr)
            continue

        lb, meta = letterbox_image(img, args.size)
        if not cv2.imwrite(str(opath), lb):
            print(f"[WARN] cannot write: {opath}", file=sys.stderr)
            continue

        meta_rows.append([
            str(ipath), str(opath),
            meta["orig_w"], meta["orig_h"], meta["out_size"],
            meta["w_new"], meta["h_new"], meta["scale"], meta["pad_x"], meta["pad_y"]
        ])
        count += 1
        if count % 500 == 0:
            print(f"[INFO] processed {count} images...")

    meta_csv = out_dir / "letterbox_meta.csv"
    with open(meta_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["orig_path","resized_path","orig_w","orig_h","out_size",
                    "w_new","h_new","scale","pad_x","pad_y"])
        w.writerows(meta_rows)

    print(f"[DONE] images: {count}")
    print(f"[DONE] meta:   {meta_csv}")
    print(f"[DONE] out:    {out_img_root}")

if __name__ == "__main__":
    main()
