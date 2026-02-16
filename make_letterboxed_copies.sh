#!/bin/bash
set -euo pipefail

# Esegue make_letterboxed_copies.py con parametri di default facilmente modificabili.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WINDOWS_CSV="mini_data_input/medicanes_new_windows.csv"
SRC_DIR="/media/fenrir/disk1/danieleda/download_EUMETSAT_data/from_gcloud"
OUT_DIR="/media/fenrir/disk1/danieleda/resized-stretched"
META_CSV="manifests/letterbox_meta.csv"
IMAGE_SIZE="224"
RESIZE_MODE="stretch" # letterbox|stretch
BUFFER_HOURS="36"
RECURSE="true"
PRESERVE_STRUCTURE="false"
WORKERS="20"
# opzionali: TIME_REGEX="..."  STRPTIME_FORMATS=("yyyy-MM-ddTHH-mm" "yyyyMMdd_HHmm")  CSV_STRPTIME_FORMATS=("yyyy-MM-dd HH:mm")

mkdir -p "$OUT_DIR"

EXTRA_ARGS=()
if [[ "${RECURSE}" == "true" ]]; then
  EXTRA_ARGS+=(--recurse)
fi
if [[ "${PRESERVE_STRUCTURE}" == "true" ]]; then
  EXTRA_ARGS+=(--preserve-structure)
fi
if [[ -n "${TIME_REGEX:-}" ]]; then
  EXTRA_ARGS+=(--time-regex "${TIME_REGEX}")
fi
if [[ -n "${STRPTIME_FORMATS+x}" ]]; then
  EXTRA_ARGS+=(--strptime "${STRPTIME_FORMATS[@]}")
fi
if [[ -n "${CSV_STRPTIME_FORMATS+x}" ]]; then
  EXTRA_ARGS+=(--csv-strptime "${CSV_STRPTIME_FORMATS[@]}")
fi

exec python scripts/make_letterboxed_copies.py \
  --windows-csv "$WINDOWS_CSV" \
  --src "$SRC_DIR" \
  --out-dir "$OUT_DIR" \
  --meta-csv "$META_CSV" \
  --image-size "$IMAGE_SIZE" \
  --resize-mode "$RESIZE_MODE" \
  --buffer-hours "$BUFFER_HOURS" \
  --workers "$WORKERS" \
  "${EXTRA_ARGS[@]}"

  #  --dry-run \    
