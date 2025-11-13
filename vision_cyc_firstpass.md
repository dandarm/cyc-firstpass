# Visione & Motivazioni — cyc‑firstpass

## 1) Executive summary
**cyc‑firstpass** è un progetto focalizzato su un obiettivo semplice e ad alto impatto:
fornire un **primo passaggio veloce** sull’intero Mediterraneo per dire *se* in un frame c’è un ciclone (presenza) e *dove* si trova il suo **centro** (stima di coordinate), producendo **ROI** (ritagli ad alta risoluzione) e un **file di predizioni** standardizzato.

Questo primo pass è deliberatamente **leggero, robusto e agnostico** rispetto ai modelli HR (High Resolution) di secondo stadio. Non è il tracker finale: è un **selettore/coarse‑locator** che accelera e stabilizza l’intera pipeline di ricerca.


## 2) Il problema che affrontiamo
- I dati satellitari del Mediterraneo sono **estesi nel tempo** e **ampli** nello spazio.
- I **cicloni mediterranei (medicanes)** sono rari, eterogenei per forma/scala e non sempre facili da distinguere da fenomeni simili.
- I modelli complessi di secondo stadio (tracking HR, analisi fine) risultano **costosi** se lanciati su tutto il bacino e su tutti i tempi.

**cyc‑firstpass** risponde a tre esigenze pratiche:
1. **Ridurre il dominio di calcolo**: passiamo al modello HR solo frame/zone promettenti.
2. **Normalizzare la geometria**: portiamo ogni frame a un formato **quadrato e omogeneo** (letterbox) senza deformazioni.
3. **Allineare le etichette temporali**: costruiamo le label da un’unica fonte autorevole di finestre temporali di “presenza ciclone”.


## 3) Perché un “first‑pass” (e non subito il modello HR)
- **Efficienza**: un classificatore/localizzatore leggero riduce drasticamente il numero di pop‑up verso lo stadio costoso.
- **Robustezza**: centralizzare la logica di labeling e normalizzazione geometrica riduce entropie e incoerenze a valle.
- **Portabilità**: la nostra uscita (CSV + ROI) è uno standard semplice da integrare con qualunque modello HR, presente o futuro.


## 4) Scelte progettuali chiave (senza codice)
**a) Letterbox offline**
- Portiamo ogni immagine full‑basin a **S×S** (tip. 384×384) preservando il **rapporto d’aspetto** e aggiungendo **padding** sui lati corti.
- Motivi: performance (evitare resize per epoca), determinismo (stesse mappe di coordinate sempre), semplificazione dei DataLoader.

**b) Labeling da finestre temporali**
- Usiamo **medicanes_new_windows.csv** come **fonte unica**: se il timestamp del frame cade in **[start_time, end_time]**, etichetta **positiva**, altrimenti **negativa**.
- Questo elimina divergenze tra più sorgenti e rende **riproducibile** lo split e le statistiche di classe.

**c) Centro come “heatmap/keypoint”**
- Rappresentiamo il centro con una **heatmap** (una “collinetta” probabilistica) da cui estraiamo il picco; è più stabile di una regressione diretta X,Y.
- Vantaggi: tollera incertezze, consente filtri di qualità (larghezza del picco), fornisce una misura intrinseca di confidenza spaziale.

**d) Output contrattualizzato**
- **preds.csv**: per ogni frame, scriviamo probabilità di presenza e coordinate del centro sia nello spazio **letterbox** (S×S) sia riportate ai **pixel originali**.
- **ROI/**: ritagli centrati sulle predizioni per alimentare il secondo stadio HR.
- Questi formati sono **stabili** e versionabili, adatti sia ad analisi manuale sia a pipeline automatiche.

## 4bis) Il modello: architettura e operatività

### Input e normalizzazione

* **Immagine di ingresso**: frame full‑basin già **letterbox** a S×S (default S=512), valori [0,1].
* **Canali**: tipicamente 1 canale (IR); estendibile a 3 canali (compositi RGB) o a stack temporali (N canali) senza cambiare il contratto I/O.
* **Normalizzazione**: per canale (media/deviaz. standard del dataset) o min‑max; l’importante è coerenza tra train/val/infer.

### Architettura (stile SimpleBaseline)

* **Backbone**: ResNet‑18/50 pre‑addestrata su ImageNet (feature compatte e robuste).
* **Decoder spaziale**: 2‑3 blocchi *deconvolution/upsampling* per riportare le feature a una griglia densa e ottenere una **heatmap** 1‑canale.
* **Head di presenza** (in parallelo): classificatore binario (logit) ottenuto da pooling globale + MLP leggero.
* **Stride di heatmap**: se l’input è S×S e lo stride effettivo è `s` (tipico `s=4`), la heatmap ha risoluzione `(S/s)×(S/s)`.

### Target e funzioni di perdita

* **Heatmap**: gaussiana 2D centrata in `(xg, yg)` (coordinate nel sistema S×S), con deviazione standard `σ` (in pixel dell’input). Loss: **MSE** tra heatmap predetta e target.
* **Presenza**: loss **BCEWithLogits** sul logit della head binaria.
* **Mascheratura**: sui **negativi** la loss heatmap è mascherata (nessun contributo), si ottimizza solo la BCE.
* **Ponderazioni**: opzionale re‑weighting positivi/negativi se il dataset è sbilanciato.

### Training (comportamento)

* **Batching** misto di positivi/negativi (rapporto controllabile via sampler o pesi).
* **Augment leggeri**: flip orizzontale (prudenza con rotazioni su dati geofisici), piccole perturbazioni di contrasto/rumore; evitare trasformazioni che alterino la geografia in modo non realistico.
* **Label “incerte”**: i frame a ridosso degli estremi finestra possono essere esclusi o marcati come *ignore* in fase di training per ridurre rumore di etichetta.

### Inferenza (decodifica e decisione)

1. **Probabilità di presenza**: `presence_prob = sigmoid(logit)`.
2. **Soglia decisionale** `τ` (tipico 0.45–0.60) per dire *c’è/non c’è* un ciclone nel frame.
3. **Decodifica centro**: picco della heatmap. Opzioni:

   * **Argmax** (semplice, veloce) → coordinate griglia `(xh, yh)`;
   * **Soft‑argmax / fit locale** (sub‑pixel) per maggiore precisione.
4. **Rimappatura**: `(xg, yg) = (xh·s, yh·s)` in pixel S×S → poi **back‑projection** ai pixel originali usando *scale* e *padding* del letterbox.
5. **ROI**: raggio `r = max(r0, k·w)` dove `w` è una misura della **larghezza del picco** (proxy d’incertezza) e `r0` una base minima; ritaglio centrato sulla predizione.

### Misure di qualità e fallback

* **Confidenza spaziale**: valore di picco della heatmap e **larghezza** del bump aiutano a filtrare falsi positivi.
* **Controlli di validità**: penalizzare picchi troppo vicini ai bordi (effetti del padding) o heatmap troppo “piatte”.
* **Top‑K (opzionale)**: in scene complesse (multi‑vortice) si possono esportare più candidati ordinati per punteggio.

### Parametri consigliati (da tarare una volta sola)

* `S=512`, `s=4` → heatmap 128×128;
* `σ` tra 6 e 10 px (riferito a S×S);
* soglia `τ` per presenza scelta su **val** per un compromesso Precision/Recall;
* `r0` (ROI base) e `k` (moltiplicatore) tarati su **val** in funzione della scala media dei cicloni.

### Casi limite e mitigazioni

* **Finestre temporali rumorose** → *ignore band* attorno agli estremi;
* **Vortici baroclini / sistemi frontali** → negativi “difficili” nel train;
* **Copertura nuvolosa irregolare** → top‑K candidati o smoothing temporale post‑hoc (fuori dallo scope di questo repo, ma compatibile con l’output).

**b) Labeling da finestre temporali**

* Usiamo **medicanes_new_windows.csv** come **fonte unica**: se il timestamp del frame cade in **[start_time, end_time]**, etichetta **positiva**, altrimenti **negativa**.
* Questo elimina divergenze tra più sorgenti e rende **riproducibile** lo split e le statistiche di classe.

**c) Centro come “heatmap/keypoint”**

* Rappresentiamo il centro con una **heatmap** (una “collinetta” probabilistica) da cui estraiamo il picco; è più stabile di una regressione diretta X,Y.
* Vantaggi: tollera incertezze, consente filtri di qualità (larghezza del picco), fornisce una misura intrinseca di confidenza spaziale.

**d) Output contrattualizzato**

* **preds.csv**: per ogni frame, scriviamo probabilità di presenza e coordinate del centro sia nello spazio **letterbox** (S×S) sia riportate ai **pixel originali**.
* **ROI/**: ritagli centrati sulle predizioni per alimentare il secondo stadio HR.
* Questi formati sono **stabili** e versionabili, adatti sia ad analisi manuale sia a pipeline automatiche.

```mermaid
flowchart TB
    I[Full-basin frame\n(letterbox S×S)] --> BN[Backbone\nResNet-18/50]
    BN --> DEC[Decoder deconv/upsampling\n(stride = s)]
    DEC --> HM[(Heatmap 1ch)]
    HM --> PK[Peak decode\n(argmax / soft-argmax)]
    PK --> XYg[(x_g, y_g) @ S×S]
    XYg --> BP[Back-projection\n→ pixel originali\n(scale, pad)]
    BP --> ROI[Crop ROI centrata]
    ROI --> OUT1[preds.csv + ROI/]

    DEC --> PH[Presence head\n(GAP + MLP → logit)]
    PH --> SG[sigmoid]
    SG --> THR{presence_prob ≥ τ?}
    THR -- sì --> ROI
    THR -- no --> OUT0[No-cyclone\n(presence_prob, no ROI)]
```

## 5) Flusso dati end‑to‑end (panoramica)
1. **Input immagini** (full‑basin, dimensioni originali, naming con data/ora inclusa).
2. **Pre‑processing**: letterbox **offline** → cartella di immagini S×S + metadati (scala, padding).
3. **Labeling**: import di `medicanes_new_windows.csv`, parsing dei timestamp dai nomi file, etichettatura positiva/negativa in base agli intervalli.
4. **(Opzionale) Keypoint dai windows**: se il CSV contiene coordinate del centro su 1290×420, le proiettiamo coerentemente nello spazio S×S.
5. **Training/validazione** del first‑pass (binario + centro come heatmap) con manifest coerenti; niente resize runtime.
6. **Inferenza**: per ogni frame → probabilità di presenza; se supera la soglia, estrazione picco heatmap → coordinate; generazione ROI; scrittura `preds.csv` + PNG.
7. **Hand‑off**: ROI + `preds.csv` diventano l’input del secondo stadio (HR), mantenendo **separazione** netta tra progetti.


## 6) Cosa entra e cosa esce
**Input minimi**
- Immagini del Mediterraneo (formato coerente e consistente nel naming: data/ora nel filename).
- `medicanes_new_windows.csv` con colonne `start_time`, `end_time` (e opzionalmente keypoint per time).

**Output principali**
- **`preds.csv`** con: percorso originale, percorso resized, `presence_prob`, coordinate del centro in S×S e riportate ai pixel originali, parametri di ritaglio ROI, path ROI.
- **`ROI/`** con i ritagli centrati.
- (Per il training) manifest `train/val/test.csv` omogenei e versionabili.


## 7) Scelte metodologiche motivate
- **Unica sorgente di verità per le label**: medicanes_new_windows → meno ambiguità, migliore governance del dataset.
- **Normalizzazione geometrica**: senza deformare le strutture nuvolose; fondamentale per stabilità di addestramento e confronto tra epoche.
- **Heatmap per il centro**: robustezza sugli outlier, possibilità di stime di incertezza (larghezza del picco) e semplici post‑filtri.
- **Separazione dei progetti**: cyc‑firstpass è indipendente per evitare accoppiamento stretto con modelli HR; i contratti di I/O sono pensati per essere longevi.


## 8) Metriche di successo
- **Presenza**: AUPRC (area sotto precision‑recall), Precision@τ, Recall@τ con τ = soglia di `presence_prob`.
- **Centro** (solo positivi): MAE/MedAE in km; % di frame con errore sotto 50/100/150 km; analisi per evento/stagione.
- **Operatività**: tempo di elaborazione per 1000 frame; % di riduzione dei frame inviati al secondo stadio; tempo medio per evento.


## 9) Assunzioni, rischi, contromisure
**Assunzioni**
- I nomi dei file contengono timestamp parsabili in modo stabile.
- Le finestre in `medicanes_new_windows.csv` sono **complete** e in formato consistente.
- Le immagini pre‑letterbox sono prodotte con la **stessa** formula di scala/padding usata per proiettare le coordinate.

**Rischi & mitigazioni**
- *Naming irregolare*: introdurre un mapping esterno `image_path,time` come fallback; validatori di parsing con report.
- *Finestre inaccurate*: loggare i frame vicini agli estremi; permettere una tolleranza configurabile (es. ±1 frame) a scopo diagnostico.
- *Errori di proiezione*: test di coerenza (round‑trip, controllo dei bordi, clipping) e report automatico di outlier.
- *Distribuzione squilibrata*: campionamento/weighting in training e split stratificato per evento.


## 10) Cosa **non** fa cyc‑firstpass
- Non effettua tracking HR o analisi morfologica fine (occhio, simmetria, ecc.).
- Non sostituisce i modelli di secondo stadio: prepara terreno e riduce il carico computazionale.
- Non dipende da uno specifico modello HR: rimane **agnostico** e interoperabile.


## 11) Interfacce “contratto” tra fasi
- **Input immagini** → devono essere accessibili e stabili (percorsi).
- **Pre‑processing** → produce S×S + CSV metadati (scala, padding).
- **Labeling** → produce manifest coerenti (train/val/test) e *opzionalmente* keypoint proiettati.
- **First‑pass** → consuma i manifest, produce `preds.csv` + ROI.
- **Secondo stadio** → consuma ROI/CSV senza richiedere conoscenza interna del first‑pass.


## 12) Governance e collaborazione
- `AGENTS.md` → guida operativa per sviluppatori/agenti (dettagli tecnici, flussi, parametri).
- `CONTRIBUTING.md` → regole minime per PR, naming branch, gestione conflitti **senza rompere `main`**.
- Documentazione **versionata** nel repo: decisioni e motivazioni restano tracciate come parte del progetto.


## 13) Roadmap (breve)
1. **Stabilizzazione labeling** da `medicanes_new_windows.csv` (test unit + smoke).
2. **Valutazione sistematica** delle metriche su stagioni/eventi; tabelle e grafici di sintesi.
3. **Top‑K candidate** (opzionale) per eventi multipli o multi‑vortice.
4. **Canali aggiuntivi** (stack temporali o RGB compositi) mantenendo la semplicità del primo pass.
5. **Integrazione CI** leggera per test di parsing/letterbox su sample data.


## 14) Glossario essenziale
- **Letterbox**: ridimensionamento a quadro S×S che preserva il rapporto d’aspetto e aggiunge padding sui lati corti (nessuna deformazione dell’immagine).
- **Heatmap** (per il centro): immagine 2D che rappresenta la probabilità di posizione del centro; il picco individua la stima.
- **ROI** (*Region of Interest*): ritaglio attorno alla predizione del centro, pensato per alimentare un secondo stadio HR.
- **Manifest**: CSV che elenca i sample da addestrare/validare/testare, con percorsi e label coerenti.

---
**In sintesi**: cyc‑firstpass è il “portiere” della pipeline per i medicanes: normalizza, decide se “far passare” un frame e indica **dove guardare**. Così il sistema complessivo diventa più rapido, più affidabile e più facile da mantenere. 

