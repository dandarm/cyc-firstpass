#!/bin/bash
#SBATCH --job-name=cyc-firstpass-train
#SBATCH --nodes=4                # aggiorna se usi un solo nodo
#SBATCH --ntasks-per-node=4      # una task per GPU
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --time=00:30:00
#SBATCH --output=cyc_first_train.out
#SBATCH --error=cyc_first.err

# set -x  # uncomment per debug
# set -euo pipefail

module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/videomae/bin/activate


# parametri principali
CONFIG="config/default.yml"
TRAIN_CSV="manifests/train.csv"
VAL_CSV="manifests/val.csv"
LOG_DIR="outputs/runs/exp_mpi_2"
mkdir -p "$LOG_DIR"

TEMPORAL_T="5"
TEMPORAL_STRIDE="6" 
BATCH_SIZE="32"
LR="3e-4"

export NCCL_DEBUG=INFO
# per debug dettagliato e gestione errori asincroni NCCL
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# imposta MASTER_ADDR/PORT per l'inizializzazione distribuita
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=12340

GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
CPUS_PER_TASK=8

mpirun --map-by socket:PE=${CPUS_PER_TASK} --report-bindings \
  python -m src.cyclone_locator.train \
    --config "$CONFIG" \
    --train_csv "$TRAIN_CSV" \
    --val_csv "$VAL_CSV" \
    --log_dir "$LOG_DIR" \
    --temporal_T "$TEMPORAL_T" \
    --temporal_stride "$TEMPORAL_STRIDE" \
    --num_workers "$CPUS_PER_TASK" \
    --dataloader_timeout_s 30
