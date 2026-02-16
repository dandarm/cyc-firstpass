# Cyclone-Center-FirstPass

First-pass **detector+locator** del centro di ciclone su **intero Mediterraneo**:
- **Presenza** (0/1) + **coordinate** (da heatmap K=1).
- Input: frame **letterbox 512x512**, no deformazioni.
- Output: CSV/JSON con (x_orig, y_orig), **probabilità di presenza**, e **ROI** ritagliate per il modello HR (VideoMAE).

## Setup

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt


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
  --images-dir data/letterboxed_512/resized \
  --out-dir path/manifests \
  --orig-size 1290 420 \
  --target-size 384 \
  --val-split 0.15 --test-split 0.15 \
  --attach-keypoints auto
```

* Legge `data/medicanes_new_windows.csv`.
* Scansiona `data/letterboxed_512/resized` (immagini già letterbox SxS).
* Scrive: `data/manifests/train.csv`, `data/manifests/val.csv`, `data/manifests/test.csv`.

Lo script etichetta ogni frame in base alle finestre `[start_time, end_time]` (inclusione chiusa).
Se il CSV contiene colonne `x_pix`,`y_pix`, vengono proiettate nella letterbox S×S e salvate come
`x_pix_resized`,`y_pix_resized` per l'uso diretto dei DataLoader.

-------------

Esempio:

image_path,presence,cx,cy<br>
/data/frames/2020-10-25T00-00.png,1,834,207 <br> 
/data/frames/2020-10-25T00-05.png,1,836,208 <br>
/data/frames/2020-10-25T00-10.png,0,,<br>






## Training
```bash
python -m src.cyclone_locator.train \
  --train_csv manifests/train.csv \
  --val_csv   manifests/val.csv \
  --image_size 384 \
  --heatmap_stride 4 \
  --heatmap_sigma_px 8 \
  --backbone resnet18 \
  --epochs 5 --bs 64 --lr 3e-4 \
  --log_dir outputs/runs/exp1
```

Backbone disponibili:

- `resnet18` / `resnet50`: early-fusion 2D, i frame temporali vengono concatenati sui canali.
- `x3d_xs` / `x3d_s` / `x3d_m`: backbone 3D puro stile X3D che mantiene la dimensione temporale fino al pooling adattivo (nessuna
  fusione temporale nei primi layer) e accetta sequenze di lunghezza configurabile (`temporal_T`, `temporal_stride`).


## Inferenza / Eval
Il nuovo entrypoint supporta sia la sola inferenza sia una modalità di valutazione completa
con curve e metriche aggregate. Esempio **eval pulita** (niente ROI):

```bash
python -m src.cyclone_locator.infer \
  --config config/default.yml \
  --checkpoint outputs/runs/exp1/checkpoints/best.ckpt \
  --manifest_csv manifests/val.csv \
  --letterbox-meta manifests/letterbox_meta.csv \
  --threshold 0.5 \
  --out_dir outputs/eval/val \
  --save-preds outputs/eval/val/preds_val.csv \
  --metrics-out outputs/eval/val/metrics_val.json \
  --sweep-curves outputs/eval/val/curves
```

Output principali:

- `preds_*.csv` con `image_path,presence_prob,x_g,y_g,logit` (+ `presence_pred` se passi `--threshold`).
- `metrics_*.json` con AUPRC, ROC-AUC, precision/recall/F1@τ, confusion matrix e metriche di localizzazione
  (MAE/MedAE/percentili entro R px, automaticamente anche in km se fornisci `--letterbox-meta`
  così da poter ricostruire le coordinate globali tramite le utility di `medicane_utils`, vedi `pixel_km_conversion.md`).
- `curves/{pr,roc}_curve.csv` se specifichi `--sweep-curves DIR`.
- Log dettagliato in `<out_dir>/eval.log`.

Per attivare la back-projection e salvare le ROI del secondo stadio:

```bash
python -m src.cyclone_locator.infer \
  --config config/default.yml \
  --checkpoint outputs/runs/exp1/checkpoints/best.ckpt \
  --manifest_csv manifests/test.csv \
  --letterbox-meta manifests/letterbox_meta.csv \
  --export-roi \
  --roi-dir outputs/eval/test/roi \
  --threshold 0.55 \
  --save-preds outputs/eval/test/preds_test.csv
```

Con `--export-roi` vengono aggiunte le colonne `x_orig,y_orig,roi_path` in `preds.csv`
e i ritagli PNG vengono salvati in `roi_dir`. Le metriche di centro rispettano la policy
"detection-first" di default; usa `--oracle-localization` per diagnostica.
