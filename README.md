# Cyclone-Center-FirstPass

First-pass **detector+locator** del centro di ciclone su **intero Mediterraneo**:
- **Presenza** (0/1) + **coordinate** (da heatmap K=1).
- Input: frame **letterbox 512x512**, no deformazioni.
- Output: CSV/JSON con (x_orig, y_orig), **probabilità di presenza**, e **ROI** ritagliate per il modello HR (VideoMAE).

## Dati attesi

Manifests CSV in `data/manifests/` con colonne:

- `image_path` — path al frame (PNG/JPG).
- `presence` — 1 se c'è un ciclone; 0 altrimenti.
- `cx`, `cy` — coordinate **in pixel originali** del centro (solo se presence=1; altrimenti vuote o -1).
- `x_pix_resized`, `y_pix_resized` — opzionali; coordinate nella griglia letterbox (se già disponibili).

## Windows labeling

Per generare nuovi manifest da un CSV di finestre temporali (`medicanes_new_windows.csv`) usa:

```bash
python scripts/make_manifest_from_windows.py \
  --windows-csv path/medicanes_new_windows.csv \
  --images-dir path/pre_letterbox_512 \
  --out-dir path/manifests \
  --orig-size 1290 420 --target-size 512
```

Lo script etichetta ogni frame in base alle finestre `[start_time, end_time]` (inclusione chiusa).
Se il CSV contiene colonne `x_pix`,`y_pix`, vengono proiettate nella letterbox S×S e salvate come
`x_pix_resized`,`y_pix_resized` per l'uso diretto dei DataLoader.

-------------

Esempio:

image_path,presence,cx,cy<br>
/data/frames/2020-10-25T00-00.png,1,834,207 <br> 
/data/frames/2020-10-25T00-05.png,1,836,208 <br>
/data/frames/2020-10-25T00-10.png,0,,<br>


## Setup

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt



## Training
python -m cyclone_locator.train --config config/default.yaml

## Inferenza
python -m cyclone_locator.infer \
  --config config/default.yaml \
  --checkpoint checkpoints/best.ckpt \
  --out_dir outputs/preds/


## Output:

preds.csv con (presence_prob, x_g,y_g in 512, x_orig,y_orig, r_crop_px, ecc.)

ROI ritagliate per il VideoMAE HR (cartella outputs/preds/roi/).