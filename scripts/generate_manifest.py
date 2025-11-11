#!/usr/bin/env python3
import argparse, csv, os, glob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_glob", required=True, help="es. /data/frames/*.png")
    ap.add_argument("--tracks_csv", default=None,
                    help="CSV con colonne: image_path,cx,cy (solo positivi); se assente => tutti negativi")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    tracks = {}
    if args.tracks_csv and os.path.exists(args.tracks_csv):
        with open(args.tracks_csv) as f:
            r = csv.DictReader(f)
            for row in r:
                tracks[os.path.abspath(row["image_path"])] = (float(row["cx"]), float(row["cy"]))

    frames = sorted(glob.glob(args.frames_glob))
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "presence", "cx", "cy"])
        for p in frames:
            apath = os.path.abspath(p)
            if apath in tracks:
                cx, cy = tracks[apath]
                w.writerow([apath, 1, cx, cy])
            else:
                w.writerow([apath, 0, "", ""])
    print("Wrote:", args.out_csv)

if __name__ == "__main__":
    main()
