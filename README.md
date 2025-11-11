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

Esempio:




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