# Cyclone First-Pass Pipeline Guide

Questa guida descrive l'architettura e il flusso operativo del modello first-pass full-basin per la rilevazione di cicloni (classificazione di presenza + regressione del centro via heatmap) e l'hand-off verso un secondo stadio ad alta risoluzione (es. VideoMAE). Il documento è pensato per programmatori e LLM-agents: offre sia una panoramica ad alto livello sia i riferimenti diretti ai file di implementazione.

## Mappa del repository
- **Config**: [`config/default.yml`](config/default.yml) contiene gli iperparametri di training e inferenza.
- **Dataset & dataloading**: [`src/cyclone_locator/datasets/med_fullbasin.py`](src/cyclone_locator/datasets/med_fullbasin.py) implementa il dataset principale `MedFullBasinDataset`. [`src/cyclone_locator/datasets/windows_labeling.py`](src/cyclone_locator/datasets/windows_labeling.py) supporta la generazione di manifest dalle finestre originali.
- **Trasformazioni**: [`src/cyclone_locator/transforms/letterbox.py`](src/cyclone_locator/transforms/letterbox.py) fornisce la logica di letterbox e le mappe di coordinate avanti/indietro.
- **Modelli**: [`src/cyclone_locator/models/simplebaseline.py`](src/cyclone_locator/models/simplebaseline.py) contiene il backbone ResNet e il decoder deconvoluzionale con le due teste (heatmap + presenza).
- **Loss**: [`src/cyclone_locator/losses/heatmap_loss.py`](src/cyclone_locator/losses/heatmap_loss.py) implementa la loss MSE condizionata sulla presenza.
- **Training / inferenza**: [`src/cyclone_locator/train.py`](src/cyclone_locator/train.py) coordina training/val con AMP opzionale; [`src/cyclone_locator/infer.py`](src/cyclone_locator/infer.py) gestisce l'esportazione di predizioni, incluso l'hand-off a stadi successivi.
- **Utility**: [`src/cyclone_locator/utils/geometry.py`](src/cyclone_locator/utils/geometry.py) e [`src/cyclone_locator/utils/metric.py`](src/cyclone_locator/utils/metric.py) coprono funzioni geometriche e metriche; [`src/cyclone_locator/utils/visualize.py`](src/cyclone_locator/utils/visualize.py) produce overlay heatmap/debug.
- **Script dati**: la cartella [`scripts/`](scripts) contiene utilità CLI per la preparazione dati, tra cui `generate_manifest.py`, `make_manifest_from_windows.py`, `letterbox_folder.py`, `make_letterboxed_copies.py`.
- **Dati di esempio**: [`mini_data_input/`](mini_data_input) ospita un manifest dimostrativo e immagini letterbox.

## Manifest CSV
Ogni campione è descritto da un record CSV con campi obbligatori:
- `image_path`: percorso (assoluto o relativo) dell'immagine.
- `presence`: etichetta binaria 0/1.
- `cx`, `cy`: coordinate del centro in pixel nello spazio originale (prima del letterbox).

Esempio:
```
image_path,presence,cx,cy
/abs/path/img_0001.png,1,423.8,287.1
/abs/path/img_0002.png,0,,
```
Se il manifest punta a immagini già letterbox, includere anche `x_pix_resized` e `y_pix_resized` (pixel nello spazio letterbox). Le colonne `cx`, `cy` restano comunque espresse nei pixel dell'immagine originale.

## Pre-processing letterbox offline
Gli script [`scripts/letterbox_folder.py`](scripts/letterbox_folder.py) e [`scripts/make_letterboxed_copies.py`](scripts/make_letterboxed_copies.py) applicano un letterbox offline per normalizzare le dimensioni input (ad es. 512×512) mantenendo il rapporto d'aspetto. Questo step migliora la coerenza del training, evita ridimensionamenti runtime costosi e permette di cache-are i metadati necessari per riportare le coordinate al dominio originale.

Per ogni immagine si salvano:
- `orig_w`, `orig_h`: dimensioni originali.
- `out_size`: lato dell'immagine letterbox.
- `scale`: fattore di scala uniforme applicato.
- `pad_x`, `pad_y`: padding sinistro/superiore (in pixel letterbox).

Formule di mappatura (mettere `scale`, `pad_x`, `pad_y` come da metadati):
```
Orig→LB:  x_lb = scale * x_orig + pad_x
          y_lb = scale * y_orig + pad_y
LB→Orig:  x_orig = (x_lb - pad_x) / scale
          y_orig = (y_lb - pad_y) / scale
```
Questi valori sono serializzati da [`scripts/generate_manifest.py`](scripts/generate_manifest.py) e riutilizzati dal dataset via CSV metadati (colonne `orig_path`, `resized_path`, `orig_w`, `orig_h`, `out_size`, `scale`, `pad_x`, `pad_y`).

## Dataset e tensori prodotti
`MedFullBasinDataset` combina il manifest e (opzionalmente) il CSV dei metadati letterbox. Durante `__getitem__`:
1. Carica l'immagine letterbox (già ridimensionata) da `resized_path` o direttamente da `image_path` se il manifest è già letterbox.
2. Mappa le coordinate `cx`, `cy` al sistema letterbox usando i metadati (funzione `_forward_map_xy`).
3. Costruisce una heatmap gaussiana 1×H×W (risoluzione `image_size / heatmap_stride`) centrata sulle coordinate letterbox divise per `heatmap_stride` con deviazione `heatmap_sigma_px / heatmap_stride`.
4. Restituisce un dizionario con:
   - `image`: tensor float32 normalizzato in [0,1], shape (C, `image_size`, `image_size`).
   - `heatmap`: tensor float32, shape (1, `image_size/stride`, `image_size/stride`).
   - `presence`: tensor float32 shape (1,).
   - Metadati (`meta_scale`, `meta_pad_x`, `meta_pad_y`, `orig_w`, `orig_h`, `image_path`) per la fase di inferenza/hand-off.

Parametri principali:
- `image_size`: lato dell'immagine letterbox (default 512).
- `heatmap_stride`: fattore di downsampling rispetto all'immagine (default 4, quindi heatmap 128×128).
- `heatmap_sigma_px`: sigma della gaussiana nello spazio pixel originale (default 8 px).
- `use_pre_letterboxed`: se `True`, richiede il CSV dei metadati per campioni non già letterbox.

## Modello, training e hand-off
Il modello `SimpleBaseline` (in [`src/cyclone_locator/models/simplebaseline.py`](src/cyclone_locator/models/simplebaseline.py)) impiega una ResNet (18 o 50, pre-addestrata su ImageNet) come backbone feature extractor. Un decoder deconvoluzionale upsample restituisce una mappa spaziale alla risoluzione della heatmap. Due teste separate producono:
- **Heatmap head**: predizione della distribuzione spaziale del centro ciclone.
- **Presence head**: probabilità logit di presenza (per frame interi).

`train.py` costruisce il modello, applica Mixed Precision (AMP) se `--amp` o `cfg.training.amp` è attivo, e ottimizza con due loss:
- `heatmap_loss`: MSE tra heatmap predetta e target, calcolata **solo** sui campioni con `presence=1` (vedi [`src/cyclone_locator/losses/heatmap_loss.py`](src/cyclone_locator/losses/heatmap_loss.py)).
- `presence_loss`: `torch.nn.functional.binary_cross_entropy_with_logits` su tutti i campioni.

La somma pesata di queste loss guida l'apprendimento con scheduler e logging definiti in `train.py`. Durante l'inferenza (`infer.py`):
1. Si calcola la heatmap e la presenza.
2. Se la presenza supera la soglia, il picco della heatmap viene riportato allo spazio originale via `LB→Orig` usando i metadati.
3. Le coordinate e il logit di presenza vengono passati al secondo stadio ad alta risoluzione (es. VideoMAE) come ROI o indici temporali.

## Estensioni e personalizzazioni
- **Backbone**: aggiungere nuove architetture in [`models/simplebaseline.py`](src/cyclone_locator/models/simplebaseline.py) o creare un file dedicato nella stessa cartella e referenziarlo in `train.py`.
- **Augmentations**: espandere `_getitem_` nel dataset o creare trasformazioni modulari in [`transforms/`](src/cyclone_locator/transforms) per ottenere pipeline più articolate.
- **Metriche**: estendere [`utils/metric.py`](src/cyclone_locator/utils/metric.py) per supportare metriche specifiche (ad es. distanza geodetica).
- **Hand-off**: la logica di esportazione verso VideoMAE o altri modelli è concentrata in [`infer.py`](src/cyclone_locator/infer.py); qui si possono aggiungere serializer personalizzati o protocolli RPC.

Seguendo i riferimenti sopra, un programmatore può modificare rapidamente singole componenti o integrarne di nuove mantenendo l'allineamento con la pipeline esistente.
