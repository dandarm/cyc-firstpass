#!/usr/bin/env python3

"""
Pre-calcola le copie **letterbox** di un'intera cartella, senza bisogno del manifest.
- Mantiene il ratio, aggiunge padding su lato corto.
- Salva le immagini ridimensionate e un CSV con le **mappe di coordinate** per tornare alle dimensioni originali.

Uso:
  python scripts/letterbox_folder.py \
    --in_dir /path/to/frames \
    --out_dir data/letterboxed/512 \
    --size 512 \
    --ext .png  # oppure .jpg

Output:
  - immagini in `out_dir/`
  - `out_dir/meta.csv` con colonne: image_path,W,H,scale,pad_x,pad_y

Nota: non altera i filename, evita ricalcoli in training.
"""
import argparse, os, glob, csv
import cv2

from cyclone_locator.transforms.letterbox import letterbox_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--ext', default='.png', help='estensione filtro: .png | .jpg | .jpeg')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.in_dir, f'*{args.ext}')))
    meta_rows = []

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE) or cv2.imread(p)
        if img is None:
            print('skip (imread fail):', p)
            continue
        lb, meta = letterbox_image(img, args.size)
        out_path = os.path.join(args.out_dir, os.path.basename(p))
        cv2.imwrite(out_path, lb)
        meta_rows.append([out_path, meta['orig_w'], meta['orig_h'], meta['scale'], meta['pad_x'], meta['pad_y']])

    meta_csv = os.path.join(args.out_dir, 'meta.csv')
    with open(meta_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_path','W','H','scale','pad_x','pad_y'])
        w.writerows(meta_rows)
    print('Wrote:', meta_csv)


if __name__ == '__main__':
    main()
