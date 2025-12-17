#!/bin/bash
set -euo pipefail

# esegue l'allenamento con i parametri e la run directory specificata nel README
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRAIN_CSV="manifests/train.csv"
VAL_CSV="manifests/val.csv"
IMAGE_SIZE="384"
HEATMAP_STRIDE="4"
HEATMAP_SIGMA="8"

# Argomenti usati dal vecchio modello 2D (early fusion): commentati per riferimento
# OLD_BACKBONE="resnet18"
# OLD_TEMPORAL_T="3"
# OLD_TEMPORAL_STRIDE="1"
# OLD_LOG_DIR="outputs/runs/exp7"

# Argomenti attivi per il nuovo modello 3D X3D
BACKBONE="x3d_xs"
TEMPORAL_T="5"
TEMPORAL_STRIDE="6"

EPOCHS="20"
BATCH_SIZE="64"
LR="3e-4"

LOG_DIR="outputs/runs/exp_x3d_temporalT${TEMPORAL_T}_stride${TEMPORAL_STRIDE}"

mkdir -p "$LOG_DIR"

# Per ripristinare il vecchio modello 2D, sostituire BACKBONE/TEMPORAL_* con gli OLD_* commentati sopra
# e, se serve, aggiornare LOG_DIR con OLD_LOG_DIR.

exec python -m src.cyclone_locator.train \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --image_size "$IMAGE_SIZE" \
  --heatmap_stride "$HEATMAP_STRIDE" \
  --heatmap_sigma_px "$HEATMAP_SIGMA" \
  --backbone "$BACKBONE" \
  --temporal_T "$TEMPORAL_T" \
  --temporal_stride "$TEMPORAL_STRIDE" \
  --log_dir "$LOG_DIR"

  #   --epochs "$EPOCHS" \
  # --bs "$BATCH_SIZE" \
  # --lr "$LR" \
