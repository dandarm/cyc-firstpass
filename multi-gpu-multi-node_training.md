# Training distribuito (multi-GPU/multi-nodo)

Questo memo spiega come lanciare l'allenamento di cyc-firstpass con PyTorch DDP su più GPU e più nodi, usando `mpirun` come launcher di default (allineato agli script Slurm del progetto VideoMAEv2).

## Implementazione nel repo
- **Scoperta risorse**: `src/cyclone_locator/utils/distributed.py` fornisce `get_resources()` che legge `RANK/LOCAL_RANK/WORLD_SIZE`, variabili OMPI e SLURM e restituisce rank globale, rank locale, world size, task per nodo e `num_workers` suggeriti. `reduce_mean` media i tensori tra i processi.
- **Training**: `src/cyclone_locator/train.py` attiva DDP automaticamente quando `world_size>1`, imposta la GPU locale con `torch.cuda.set_device(local_rank)`, inizializza `torch.distributed.init_process_group(backend="nccl", init_method="env://")`, usa `DistributedSampler` per il train loader e incapsula il modello in `DistributedDataParallel`. Log, valutazione e checkpoint vengono eseguiti solo su rank 0; le metriche di train sono mediate su tutti i processi. Batch size per GPU è `train.batch_size`; LR viene moltiplicato per `world_size` se `scale_lr_by_world_size=true`.
- **Config**: `config/default.yml` interpreta `train.batch_size` come batch per GPU (batch globale = batch_size * world_size) e offre `train.scale_lr_by_world_size` per scalare il learning rate.

## Comandi esempio (mpirun)
- **Single-node, 4 GPU**:
  ```bash
  mpirun -np 4 python -m src.cyclone_locator.train \
    --config config/default.yml \
    --train_csv manifests/train.csv \
    --val_csv manifests/val.csv \
    --log_dir outputs/runs/exp_ddp_mpi
  ```
- **Multi-nodo (Slurm 2 nodi × 4 GPU)**:
  ```bash
  #!/bin/bash
  #SBATCH --nodes=2
  #SBATCH --ntasks-per-node=4
  #SBATCH --gres=gpu:4
  #SBATCH --cpus-per-task=10
  #SBATCH --partition=boost_usr_prod
  #SBATCH --time=04:00:00
  export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
  export MASTER_PORT=29500
  mpirun --map-by socket:PE=$SLURM_CPUS_PER_TASK --report-bindings \
    python -m src.cyclone_locator.train \
      --config config/default.yml \
      --train_csv manifests/train.csv \
      --val_csv manifests/val.csv \
      --log_dir outputs/runs/exp_ddp_mpi
  ```

## Note operative
- `mpirun` popola le variabili `OMPI_COMM_WORLD_*` lette da `get_resources()` per determinare rank e world size.
- Assicurati che `MASTER_ADDR` e `MASTER_PORT` siano esportate nel contesto del job (Slurm lo fa nello script sopra).
- Val/test girano solo su rank 0 per evitare duplicazioni; i checkpoint vengono salvati una sola volta.
- I worker dei dataloader vengono limitati a `min(cfg.train.num_workers, SLURM_CPUS_PER_TASK)` se la variabile SLURM è presente.
