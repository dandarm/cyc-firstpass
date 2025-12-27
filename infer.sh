#!/bin/bash
set -euo pipefail

# esegue l'inferenza con i parametri suggeriti nel README
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_PATH="config/default.yml"
RUN_DIR="outputs/runs/exp_x3dm_dsnt_6_notempstride"
OUT_DIR="$RUN_DIR/preds"
LETTERBOX_META="manifests/letterbox_meta.csv"
METRICS_OUT="$OUT_DIR/metrics.json"
SWEEP_CURVES_DIR="$OUT_DIR/curves"
ROI_DIR="$OUT_DIR/roi"
EXPORT_ROI="false"
CENTER_THRESHOLDS_KM=""
PRESENCE_THRESHOLD="0.5"
ROI_BASE_RADIUS_PX="128"
ROI_SIGMA_MULTIPLIER="2.0"
PRESENCE_FROM_PEAK="true"
BACKBONE=""              # se vuoto usa config.train.backbone
PEAK_THRESHOLD=""
PEAK_POOL=""             # se vuoto usa infer.peak_pool
PEAK_TAU=""              # se vuoto usa infer.peak_tau
SOFT_ARGMAX="true"       # consigliato per modelli DSNT
SOFT_ARGMAX_TAU=""       # se vuoto usa infer.center_tau (o loss.dsnt_tau fallback)

CHECKPOINT_PATH="$RUN_DIR/best.ckpt"
MANIFEST_CSV="manifests/test.csv"
SAVE_PREDS="$RUN_DIR/preds_test.csv"

mkdir -p "$OUT_DIR"
mkdir -p "$SWEEP_CURVES_DIR"
if [[ "$EXPORT_ROI" == "true" ]]; then
  mkdir -p "$ROI_DIR"
fi

EXTRA_ARGS=()
if [[ "$EXPORT_ROI" == "true" ]]; then
  EXTRA_ARGS+=(--export-roi --roi-dir "$ROI_DIR")
fi
if [[ -n "$CENTER_THRESHOLDS_KM" ]]; then
  read -r -a _km_thr <<<"$CENTER_THRESHOLDS_KM"
  EXTRA_ARGS+=(--center-thresholds-km "${_km_thr[@]}")
fi

exec python -m src.cyclone_locator.infer \
  --config "$CONFIG_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --out_dir "$OUT_DIR" \
  --manifest_csv "$MANIFEST_CSV" \
  --letterbox-meta "$LETTERBOX_META" \
  --threshold "$PRESENCE_THRESHOLD" \
  $( [[ -n "$BACKBONE" ]] && echo "--backbone $BACKBONE" ) \
  $( [[ "$PRESENCE_FROM_PEAK" == "true" ]] && echo "--presence-from-peak" ) \
  $( [[ -n "$PEAK_THRESHOLD" ]] && echo "--peak-threshold $PEAK_THRESHOLD" ) \
  $( [[ -n "$PEAK_POOL" ]] && echo "--peak-pool $PEAK_POOL" ) \
  $( [[ -n "$PEAK_TAU" ]] && echo "--peak-tau $PEAK_TAU" ) \
  $( [[ "$SOFT_ARGMAX" == "true" ]] && echo "--soft-argmax" ) \
  $( [[ -n "$SOFT_ARGMAX_TAU" ]] && echo "--soft-argmax-tau $SOFT_ARGMAX_TAU" ) \
  --save-preds "$SAVE_PREDS" \
  --metrics-out "$METRICS_OUT" \
  --sweep-curves "$SWEEP_CURVES_DIR" \
  --roi-base-radius "$ROI_BASE_RADIUS_PX" \
  --roi-sigma-multiplier "$ROI_SIGMA_MULTIPLIER" \
  "${EXTRA_ARGS[@]}"
