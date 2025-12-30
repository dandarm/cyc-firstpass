#!/bin/bash
#set -euo pipefail

# Esegue l'allenamento con stacking temporale (early fusion) passando temporal_T
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="config/default.yml"
TRAIN_CSV="manifests/train.csv"
VAL_CSV="manifests/val.csv"

HEATMAP_STRIDE="1"
HEATMAP_SIGMA="9"

#LOG_DIR="outputs/runs/exp13_temporalT${TEMPORAL_T}_bs${BATCH_SIZE}_buffer36_stride6"
LOG_DIR="outputs/runs/exp_9_hm-upsample+conv_stride1"
mkdir -p "$LOG_DIR"


HEATMAP_NEG_MULT="0.7"
HEATMAP_POS_MULT="2.5"
DSNT_TAU="0.7"            # temperatura per softmax2D (DSNT)
DSNT_COORD_LOSS="l1"      # l1|l2
PEAK_TAU="0.5"            # tau per logsumexp (più piccolo -> più simile a max)



BATCH_SIZE="40"

exec python -m src.cyclone_locator.train \
  --config "$CONFIG" \
  --train_csv "$TRAIN_CSV" \
  --val_csv "$VAL_CSV" \
  --log_dir "$LOG_DIR" \
  --dsnt_coord_loss "$DSNT_COORD_LOSS" \
  --heatmap_stride "$HEATMAP_STRIDE" \
  --heatmap_sigma_px "$HEATMAP_SIGMA" \
  --heatmap_neg_multiplier "$HEATMAP_NEG_MULT" \
  --heatmap_pos_multiplier "$HEATMAP_POS_MULT" \
  --bs "$BATCH_SIZE" 

 --peak_tau "$PEAK_TAU" \
--dsnt_tau "$DSNT_TAU" \

