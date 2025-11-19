#!/bin/bash
set -euo pipefail

# esegue l'inferenza con i parametri suggeriti nel README
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_PATH="config/default.yml"
CHECKPOINT_PATH="outputs/runs/exp1/best.ckpt"
OUT_DIR="outputs/preds"
MANIFEST_CSV="manifests/test.csv"
LETTERBOX_META="manifests/letterbox_meta.csv"
SAVE_PREDS="$OUT_DIR/preds.csv"
METRICS_OUT="$OUT_DIR/metrics.json"
SWEEP_CURVES_DIR="$OUT_DIR/curves"
ROI_DIR="$OUT_DIR/roi"
PRESENCE_THRESHOLD="0.5"
ROI_BASE_RADIUS_PX="128"
ROI_SIGMA_MULTIPLIER="2.0"

mkdir -p "$OUT_DIR"
mkdir -p "$SWEEP_CURVES_DIR"

exec python -m src.cyclone_locator.infer \
  --config "$CONFIG_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --out_dir "$OUT_DIR" \
  --manifest_csv "$MANIFEST_CSV" \
  --letterbox-meta "$LETTERBOX_META" \
  --threshold "$PRESENCE_THRESHOLD" \
  --save-preds "$SAVE_PREDS" \
  --metrics-out "$METRICS_OUT" \
  --sweep-curves "$SWEEP_CURVES_DIR" \
  --roi-base-radius "$ROI_BASE_RADIUS_PX" \
  --roi-sigma-multiplier "$ROI_SIGMA_MULTIPLIER"

  #--export-roi \
  #--roi-dir "$ROI_DIR" \
