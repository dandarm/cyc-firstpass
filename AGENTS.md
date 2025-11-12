# AGENTS.md — Cyclone-Center-FirstPass

Questa guida è pensata per **agenti AI** e sviluppatori. Descrive il flusso end‑to‑end, dove trovare il codice, e come orchestrare i job.

## Obiettivo
Pipeline **two‑stage** per full‑basin: (A) *first‑pass locator* rapido (heatmap+presence), (B) *HR tracker* (VideoMAE) sulle ROI ritagliate.

---
## 0) Struttura repository

````
cyc-firstpass/
├─ src/cyclone_locator/
│  ├─ datasets/med_fullbasin.py         # Dataset on‑the‑fly da manifest (letterbox al volo)
│  ├─ models/simplebaseline.py          # ResNet+deconv → heatmap(1ch) + presenza(1)
│  ├─ transforms/letterbox.py           # Resize con padding, mappe forward/inverse
│  └─ utils/{geometry.py,metrics.py,visualize.py}
├─ scripts/
│  ├─ generate_manifest.py              # Costruisce manifest [image_path,presence,cx,cy]
│  ├─ make_letterboxed_copies.py        # Pre‑letterbox da manifest → immagini + manifest_letterboxed.csv
│  └─ letterbox_folder.py               # **NUOVO**: pre‑letterbox di una cartella senza manifest
├─ config/default.yml                   # Config training/inferenza
├─ requirements.txt | pyproject.toml
└─ README.md | AGENTS.md
````

---
## 1) Pre‑processing a scelta (consigliato: pre‑letterbox offline)
**Per velocità**, evitare resize ripetuti in training. Due opzioni equivalenti per agenti:

### 1.a) Se hai già un *manifest* (e opzionalmente coord. centro)
- Script: `scripts/make_letterboxed_copies.py`
- Input: `--manifest_csv data/manifests/train.csv`
- Output: cartella di immagini 512×512 + `manifest_letterboxed.csv` (contiene mapping `x_g,y_g` coerenti).

### 1.b) Se NON hai un manifest (solo una cartella di frame)
- Script: `scripts/letterbox_folder.py` (nuovo).
- Input: `--in_dir /path/frames` `--size 512` `--ext .png`
- Output: `out_dir/*.png` + `out_dir/meta.csv` con colonne `W,H,scale,pad_x,pad_y`.
- Se poi vuoi allenare con centri noti, unisci `meta.csv` con un CSV di *tracks* (cx,cy in pixel originali) e calcola `x_g,y_g` usando `transforms.forward_map_xy`.

---
## 2) Dataset & Training (first‑pass)
- Modulo: `src/cyclone_locator/datasets/med_fullbasin.py`
- Modello: `src/cyclone_locator/models/simplebaseline.py` (ResNet18 default)
- Loss: `HeatmapMSE` (solo positivi) + `BCEWithLogits` per presenza.
- Config: `config/default.yml` → chiavi `train.*`, `loss.*`, `data.*`.
- Avvio: `python -m cyclone_locator.train --config config/default.yml`

**Parametri chiave**
- `train.image_size` (default 512), `train.heatmap_stride` (default 4 → output 128×128).
- `loss.heatmap_sigma_px` (default 8) controlla spread del target.
- `train.backbone` "resnet18|resnet50" (IMAGENET pre‑weights).

**Note per agenti**
- Se usi immagini pre‑letterbox: puoi patchare il Dataset per leggere già i 512×512 evitando chiamate a `letterbox_image` (micro‑ottimizzazione).

---
## 3) Inferenza + ROI emit per HR tracker
- Script: `python -m cyclone_locator.infer --config config/default.yml --checkpoint checkpoints/best.ckpt`
- Output: `outputs/preds/preds.csv` con colonne: `image_path,presence_prob,x_g,y_g,x_orig,y_orig,r_crop_px,roi_path`.
- Logica ROI: raggio `r = max(roi_base_radius_px, k*sigma_est)` dove `sigma_est` deriva dalla larghezza della heatmap (`utils.metrics.peak_and_width`).
- Hook: integra qui la chiamata al tuo **VideoMAE** ad alta risoluzione passando la cartella `outputs/preds/roi/`.

---
## 4) Metriche e check rapidi
- Funzioni in `src/cyclone_locator/utils/metrics.py`
  - `mae_km(...)`, `pct_within_Rkm(...)`, `peak_and_width(...)`.
- Per QA visivo: `utils/visualize.overlay_heatmap` per disegnare la heatmap sul frame.

---
## 5) Job graph per un agente
1. **Prep**: se necessario `scripts/letterbox_folder.py` → `meta.csv` (idempotente).
2. **Manifest**: `scripts/generate_manifest.py` per costruire *train/val/test*.
3. **Train**: `python -m cyclone_locator.train ...`.
4. **Infer**: `python -m cyclone_locator.infer ...`.
5. **Hand‑off**: passa ROI al pipeline **VideoMAE HR**; salva embedding, update del tracker.
6. **Report**: calcola metriche; pubblica `preds.csv` e figure di overlay.

---
## 6) Decisioni progettuali (per perché così)
- **Letterbox offline** per evitare CPU‑bound in loop di training.
- **Heatmap+presence** compatibile con *positivi/negativi*; evita soglie ad‑hoc.
- **ResNet18** come default per *time‑to‑first‑result*; switch facile a ResNet50.
- **Stride=4**: buon compromesso risoluzione/compute per localizzazione di un singolo picco.

---
## 7) Estensioni previste
- Multi‑canale (es. stack temporali) ⇒ aumenta `in_ch` nello stem e normalizza.
- Positional prior (maschera mare/terra) come canale extra.
- Sostituire backbone con **SatMAE/Prithvi** (se disponibile via timm) per domain adaptation satellitare.

---
## 8) Glossario minimo
- **Letterbox**: resize che non deforma l'immagine; aggiunge *padding* nero sui lati corti.
- **Heatmap**: immagine 2D che codifica la probabilità di posizione del centro con un "bump" gaussiano.
- **ROI**: *region of interest*, ritaglio ad alta risoluzione centrato sulla predizione.
- **Presence head**: classificatore binario che dice se un ciclone è presente o no nel frame.
