#!/bin/bash
#SBATCH --job-name=cyc-firstpass-train
#SBATCH --nodes=4                # aggiorna se usi un solo nodo
#SBATCH --ntasks-per-node=4      # una task per GPU
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH --time=02:40:00
#SBATCH --output=cyc_first_train.out
#SBATCH --error=cyc_first.err
#SBATCH --exclude=lrdn0495        # evita nodo con GPU in errore ECC (vedi cyc_first.err: rank 5)

# set -x  # uncomment per debug
# set -euo pipefail

module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/videomae/bin/activate

export PYTHONUNBUFFERED=1

# parametri principali
CONFIG="config/default.yml"
TRAIN_CSV="manifests/train.csv"
VAL_CSV="manifests/val.csv"
LOG_DIR="outputs/runs/exp_7_notempstride_hmupsample"
mkdir -p "$LOG_DIR"

TEMPORAL_T="16"
TEMPORAL_STRIDE="1" 

# Argomenti attivi per il nuovo modello 3D X3D
BACKBONE="x3d_m"  #"x3d_xs"
HEATMAP_NEG_MULT="0.7"
HEATMAP_POS_MULT="1.2"
HEATMAP_LOSS="dsnt"      # mse|focal|dsnt
DSNT_TAU="0.7"            # temperatura per softmax2D (DSNT)
DSNT_COORD_LOSS="l1"      # l1|l2
PEAK_TAU="0.5"            # tau per logsumexp (più piccolo -> più simile a max)

#export NCCL_DEBUG=INFO
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
    --heatmap_loss "$HEATMAP_LOSS" \
    --dsnt_tau "$DSNT_TAU" \
    --dsnt_coord_loss "$DSNT_COORD_LOSS" \
    --peak_tau "$PEAK_TAU" \
    --backbone "$BACKBONE" \
    --temporal_T "$TEMPORAL_T" \
    --temporal_stride "$TEMPORAL_STRIDE" \
    --num_workers "$CPUS_PER_TASK" \
    --dataloader_timeout_s 130 \
    --heatmap_neg_multiplier "$HEATMAP_NEG_MULT" \
    --heatmap_pos_multiplier "$HEATMAP_POS_MULT"
