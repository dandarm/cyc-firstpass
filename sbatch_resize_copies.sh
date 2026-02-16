#!/bin/bash
#SBATCH --job-name=cyc-resize-copies
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=boost_usr_prod
#SBATCH --time=00:30:00
#SBATCH --output=resize_copies.out
#SBATCH --error=resize_copies.err

#set -euo pipefail

module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/videomae/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Nota: nei job Slurm lo script viene copiato in `/var/spool/...`, quindi
# `BASH_SOURCE[0]` non punta al repo. Usiamo la submit dir come root del repo.
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
if [[ ! -d "scripts" ]]; then
  echo "ERROR: directory 'scripts/' non trovata in $(pwd). Lancia 'sbatch' dalla root del repo (o imposta SLURM_SUBMIT_DIR correttamente)." >&2
  exit 2
fi

WINDOWS_CSV="mini_data_input/medicanes_new_windows.csv"
SRC_DIR="/leonardo_scratch/fast/IscrC_SAME-D/Medicanes_Data/from_gcloud"
OUT_DIR="/leonardo_scratch/fast/IscrC_SAME-D/Medicanes_Data/from_gcloud/resized-stretched"
META_CSV="manifests/letterbox_meta.csv"
IMAGE_SIZE="224"
RESIZE_MODE="stretch" # letterbox|stretch
BUFFER_HOURS="96"
RECURSE="true"
PRESERVE_STRUCTURE="false"
WORKERS="${SLURM_CPUS_PER_TASK:-20}"
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

srun python -u scripts/make_letterboxed_copies.py \
  --windows-csv "$WINDOWS_CSV" \
  --src "$SRC_DIR" \
  --out-dir "$OUT_DIR" \
  --meta-csv "$META_CSV" \
  --image-size "$IMAGE_SIZE" \
  --resize-mode "$RESIZE_MODE" \
  --buffer-hours "$BUFFER_HOURS" \
  --workers "$WORKERS" \
  "${EXTRA_ARGS[@]}"
