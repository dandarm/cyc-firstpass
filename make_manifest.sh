#!/bin/bash
set -euo pipefail

# Esegue lo script di generazione manifest secondo l'esempio del README
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WINDOWS_CSV="mini_data_input/medicanes_new_windows.csv"
IMAGES_DIR="/media/fenrir/disk1/danieleda/resized" 
OUT_DIR="manifests"
ORIG_SIZE_X="1290"
ORIG_SIZE_Y="420"
TARGET_SIZE="384"
VAL_SPLIT="0.15"
TEST_SPLIT="0.15"
ATTACH_KEYPOINTS="auto"
BUFFER_HOURS="36"

mkdir -p "$OUT_DIR"

exec python scripts/make_manifest_from_windows.py \
  --windows-csv "$WINDOWS_CSV" \
  --images-dir "$IMAGES_DIR" \
  --out-dir "$OUT_DIR" \
  --orig-size "$ORIG_SIZE_X" "$ORIG_SIZE_Y" \
  --target-size "$TARGET_SIZE" \
  --val-split "$VAL_SPLIT" --test-split "$TEST_SPLIT" \
  --attach-keypoints "$ATTACH_KEYPOINTS" \
  --buffer-hours "$BUFFER_HOURS"
