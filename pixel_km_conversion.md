# Conversione coordinate pixel ↔ chilometri

Questo documento raccoglie tutti i punti del codice che gestiscono il passaggio fra coordinate in pixel, coordinate geografiche (lat/lon) e distanze in chilometri. È pensato come guida operativa: ogni sezione indica il file e le righe da consultare per replicare la procedura.

## Riferimenti chiave
- `medicane_utils/geo_const.py:4-90` – definisce il dominio geografico, crea l'oggetto Basemap, espone `get_lon_lat_grid_2_pixel` e la funzione vettoriale `get_cyclone_center_pixel_vector` (approccio raccomandato). Le vecchie utility a scala lineare restano solo come `_old_*` per retrocompatibilità.
- `engine_for_tracking.py:23-90` – implementa il percorso inverso: `_pixels_to_latlon` associa a ogni coppia (x, y) globale la lat/lon corrispondente tramite la stessa griglia usata da `get_cyclone_center_pixel_vector`, mentre `_haversine_km` e `batch_geo_distance_km` trasformano coordinate geografiche in distanze chilometriche.
- `inference_tracking.py:37-114` – esempio concreto di conversione per singolo campione (`_compute_sample_record`), utile come blueprint per calcolare errori sia in pixel sia in chilometri.

## 1. Preparare la mappa lat/lon dei pixel

1. `medicane_utils/geo_const.py:9-45` crea un oggetto Basemap geostazionario centrato sul Mediterraneo e usa il metodo `makegrid(image_w=1290, image_h=420)` per generare `lon_grid` e `lat_grid`.
2. `get_lon_lat_grid_2_pixel` restituisce queste matrici di corrispondenza (più le coordinate metriche X/Y) e viene richiamata sia lato dataset sia lato tracking per avere una lookup table consistente.


## 2. Da chilometri/lat-lon a pixel

Quando si parte da coordinate fisiche (chilometri o lat/lon) e si vuole sapere a che pixel corrispondono:

1. Usa `get_cyclone_center_pixel_vector` (`medicane_utils/geo_const.py:51-84`):  
   - genera o riusa la griglia `lon_grid`/`lat_grid` tramite `get_lon_lat_grid_2_pixel`;  
   - appiattisce la griglia e costruisce un `cKDTree` per associare ogni coppia (lon, lat) al pixel più vicino;  
   - restituisce `x_pix` e `y_pix` già corretti con il ribaltamento verticale (`image_h - i`), pronti per essere confrontati con le tile.
   - Questa funzione è vettoriale ed è quella impiegata nei notebook di preprocessing (`Analyze_Manos_tracks.ipynb`) e nei builder moderni.
2. Le vecchie funzioni lineari ` _old_compute_pixel_scale`, `_old_coord2px`, `_old_get_cyclone_center_pixel` e `_old_inside_tile` (`dataset/build_dataset.py:117-190`) restano nel codice solo come riferimento storico: calcolavano una scala px/metri proiettando i corner e applicavano una trasformazione affine, ma non sono più usate. Mantenerle con prefisso `_old_` chiarisce che non devono essere richiamate in nuove pipeline.
3. `inside_tile_faster` (`dataset/build_dataset.py:208-215`) continua a lavorare su coordinate pixel già proiettate: se utilizzi `get_cyclone_center_pixel_vector` o conversioni equivalenti, puoi passarle direttamente a questa funzione per verificare l'appartenenza a una tile 224×224.

## 3. Da pixel a lat/lon

Il tracking lavora su coordinate pixel relative alla tile. Per riportarle nel sistema geografico:

1. `engine_for_tracking._get_lon_lat_grid` (`engine_for_tracking.py:23-27`) richiama `get_lon_lat_grid_2_pixel` e memorizza la griglia (cache LRU per non ricomputare).
2. `_pixels_to_latlon` (`engine_for_tracking.py:29-36`) arrotonda e clippa le coordinate pixel globali, inverte l'asse Y (`row_idx = IMAGE_HEIGHT - 1 - y_idx`) e indicizza `lat_grid` / `lon_grid` per ottenere le coordinate geografiche corrispondenti.
3. `_parse_tile_offsets` (`engine_for_tracking.py:51-56`) legge gli offset (x, y) dal nome della cartella/tile. Le coordinate predette da rete e ground-truth sono relative alla tile, quindi `batch_geo_distance_km` somma questi offset per ottenere posizioni globali prima di passare `_pixels_to_latlon`.

### Coerenza con `get_cyclone_center_pixel_vector`
Il percorso “lat/lon → pixel” di `get_cyclone_center_pixel_vector` e quello inverso “pixel → lat/lon” di `_pixels_to_latlon` sono coerenti perché:
- entrambi usano la stessa `lon_grid`/`lat_grid` prodotta da `get_lon_lat_grid_2_pixel`;
- entrambi applicano il ribaltamento sull’asse Y (`image_h - i` in `get_cyclone_center_pixel_vector`, `row_idx = IMAGE_HEIGHT - 1 - y_idx` in `_pixels_to_latlon`);
- l’unica differenza pratica è il verso dell’operazione (nearest-neighbour search tramite KDTree vs indexing diretto), quindi una coppia convertita avanti e indietro può differire solo per arrotondamenti al pixel più vicino.

## 4. Da pixel a chilometri

Una volta ottenute lat/lon per predizione e target:

1. `_haversine_km` (`engine_for_tracking.py:39-48`) applica la formula dell’haversine con `EARTH_RADIUS_KM = 6371.0088` (raggio medio terrestre in km). La funzione:
   - converte latitudini e longitudini in radianti (`np.radians`);
   - calcola le differenze `dlat` e `dlon`;
   - costruisce il termine `a = sin²(dlat/2) + cos(lat1)*cos(lat2)*sin²(dlon/2)`;
   - ottiene l’angolo centrale `c = 2*arcsin(sqrt(a))`;
   - restituisce la distanza sferica `EARTH_RADIUS_KM * c`.  
   È quindi un calcolo geodetico standard su sfera che prende in input array NumPy (anche vettoriali) e produce distanze in km.
2. `batch_geo_distance_km` (`engine_for_tracking.py:59-90`) mette tutto insieme:
   - converte tensori PyTorch in NumPy (`pred_np`, `target_np`);
   - ricava gli offset con `_parse_tile_offsets` e calcola le coordinate globali (`global_pred`, `global_target`);
   - richiama `_pixels_to_latlon` per ottenere lat/lon assoluti;
   - calcola la distanza media in chilometri con `_haversine_km`, scartando eventuali valori non finiti.
3. Durante l’inferenza, `_compute_sample_record` in `inference_tracking.py:37-114` replica gli stessi passi per ogni sample e salva sia l’errore in pixel (`err_px`) sia quello in chilometri (`err_km`), utile come riferimento pratico.

## 5. Procedure operative

### 5.1 Calcolare quanti pixel corrispondono a un offset in chilometri
Quando possibile usa `get_cyclone_center_pixel_vector` per convertire direttamente lat/lon note in pixel. Se devi ancora passare da una distanza metrica a pixel senza conoscere le coordinate finali (es. offset teorico lungo un asse), puoi riferirti alle funzioni `_old_compute_pixel_scale` e `_old_coord2px` (`dataset/build_dataset.py:117-160`) per capire come ricavare i fattori `px/metri`, ma ricorda che sono deprecate e non più richiamate nelle pipeline attive.

### 5.2 Convertire una coppia pixel globali in lat/lon e distanza
1. Somma gli offset della tile alle coordinate relative (vedi `_parse_tile_offsets` e `global_pred = pred_np + offsets` in `engine_for_tracking.py:74-83`).
2. Passa la coppia risultante a `_pixels_to_latlon` (`engine_for_tracking.py:29-36`) per ottenere latitudine e longitudine.
3. Se devi confrontare due punti, invia entrambe le coppie lat/lon a `_haversine_km` e, se necessario, riusa `batch_geo_distance_km` per gestire vettori interi.

### 5.3 Pipeline end-to-end (pixel ↔ km)
Per misurare l’errore del tracker in chilometri:
1. Carica le predizioni e i target (in pixel relativi alla tile).
2. Ricava gli offset dai nomi dei video/tile (`engine_for_tracking.py:51-56`) e ottieni le coordinate globali.
3. Mappa i pixel globali in lat/lon con `_pixels_to_latlon`.
4. Usa `_haversine_km` per ottenere la distanza in km. Se vuoi l’errore medio del batch, affidati direttamente a `batch_geo_distance_km`.

## 6. Note e best practice
- La griglia lat/lon è valida solo per immagini 1290×420; se cambiano dimensioni occorre rigenerarla chiamando `get_lon_lat_grid_2_pixel` con i nuovi parametri sia lato dataset che lato tracking.
- Quando lavori con offset manuali o affini, assicurati di invertire l’asse Y nello stesso modo in cui fanno `coord2px` e `_pixels_to_latlon`, altrimenti il punto risulterà ribaltato verticalmente.
- Per debug veloci è spesso sufficiente usare `_compute_sample_record` (`inference_tracking.py:37-114`), che incapsula tutti i passaggi e restituisce sia coordinate in pixel che lat/lon, oltre a `err_km`.
- Se devi processare molte coordinate lat/lon simultaneamente (es. durante il preprocessing dei CSV Manos), usa la funzione vettoriale `get_cyclone_center_pixel_vector` (cfr. `medicane_utils/geo_const.py:51-84` e note in `docs/Analyze_Manos_tracks.md`), così resti allineato alla trasformazione usata anche nel tracking.

Seguendo i riferimenti indicati sopra è possibile implementare o verificare qualsiasi conversione pixel ↔ km mantenendo allineata la pipeline con il codice di riferimento del repository.
