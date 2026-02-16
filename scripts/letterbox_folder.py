#!/usr/bin/env python3
# scripts/letterbox_folder.py
import argparse, sys, csv, pathlib
import cv2

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from cyclone_locator.transforms.letterbox import resize_image

def iter_images(root, exts):
    root = pathlib.Path(root)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def main():
    ap = argparse.ArgumentParser(description="Pre-compute letterbox for a whole folder")
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--size", type=int, default=384, help="output square size, e.g. 512/448/320/…")
    ap.add_argument(
        "--resize-mode",
        choices=["letterbox", "stretch"],
        default="letterbox",
        help="Modalità resize: letterbox (AR + padding) oppure stretch (distorsione senza padding).",
    )
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

        lb, meta = resize_image(img, args.size, mode=args.resize_mode)
        if not cv2.imwrite(str(opath), lb):
            print(f"[WARN] cannot write: {opath}", file=sys.stderr)
            continue

        meta_rows.append([
            str(ipath), str(opath),
            meta["orig_w"], meta["orig_h"], args.size,
            meta["w_new"], meta["h_new"], meta["scale"],
            meta.get("scale_x", meta["scale"]), meta.get("scale_y", meta["scale"]),
            meta["pad_x"], meta["pad_y"], args.resize_mode
        ])
        count += 1
        if count % 500 == 0:
            print(f"[INFO] processed {count} images...")

    meta_csv = out_dir / "letterbox_meta.csv"
    with open(meta_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["orig_path","resized_path","orig_w","orig_h","out_size",
                    "w_new","h_new","scale","scale_x","scale_y","pad_x","pad_y","resize_mode"])
        w.writerows(meta_rows)

    print(f"[DONE] images: {count}")
    print(f"[DONE] meta:   {meta_csv}")
    print(f"[DONE] out:    {out_img_root}")

if __name__ == "__main__":
    main()
