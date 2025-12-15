#!/bin/bash
#SBATCH --job-name=cyc-firstpass-infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --time=00:20:00
#SBATCH --output=cyc_first_infer.out
#SBATCH --error=cyc_first_infer.err

#set -euo pipefail

module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/videomae/bin/activate

export PYTHONUNBUFFERED=1

CONFIG_PATH="config/default.yml"
OUT_DIR="outputs/preds"
LETTERBOX_META="manifests/letterbox_meta.csv"
METRICS_OUT="$OUT_DIR/metrics.json"
SWEEP_CURVES_DIR="$OUT_DIR/curves"
ROI_DIR="$OUT_DIR/roi"
EXPORT_ROI="false"
CENTER_THRESHOLDS_KM=""
PRESENCE_THRESHOLD="0.5"
ROI_BASE_RADIUS_PX="128"
ROI_SIGMA_MULTIPLIER="2.0"

CHECKPOINT_PATH="outputs/runs/exp_mpi_13/best.ckpt"
MANIFEST_CSV="manifests/test.csv"
SAVE_PREDS="outputs/runs/exp_mpi_13/preds_test.csv"

mkdir -p "$OUT_DIR" "$SWEEP_CURVES_DIR"
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

python -u -m src.cyclone_locator.infer \
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
  --roi-sigma-multiplier "$ROI_SIGMA_MULTIPLIER" \
  "${EXTRA_ARGS[@]}"
