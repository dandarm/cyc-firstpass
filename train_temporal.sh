#!/bin/bash
set -euo pipefail

# Esegue l'allenamento con stacking temporale (early fusion) passando temporal_T
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRAIN_CSV="manifests/train.csv"
VAL_CSV="manifests/val.csv"
IMAGE_SIZE="384"
HEATMAP_STRIDE="4"
HEATMAP_SIGMA="8"
BACKBONE="resnet18"
TEMPORAL_T="3"

EPOCHS="20"
BATCH_SIZE="64"
LR="3e-4"

LOG_DIR="outputs/runs/exp7_temporalT${TEMPORAL_T}"

mkdir -p "$LOG_DIR"

exec python -m src.cyclone_locator.train \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --image_size "$IMAGE_SIZE" \
  --heatmap_stride "$HEATMAP_STRIDE" \
  --heatmap_sigma_px "$HEATMAP_SIGMA" \
  --backbone "$BACKBONE" \
  --temporal_T "$TEMPORAL_T" \
  --log_dir "$LOG_DIR"

  #   --epochs "$EPOCHS" \
  # --bs "$BATCH_SIZE" \
  # --lr "$LR" \
