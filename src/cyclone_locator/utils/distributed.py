import os
from typing import Optional, Tuple

import torch.distributed as dist


def get_resources() -> Tuple[int, int, int, int, int]:
    """Autodetect launch context (torchrun/mpirun/srun) and return ranks/world size."""
    rank = 0
    local_rank = 0
    world_size = 1
    local_size = 1

    if os.environ.get("RANK"):
        # launched with torchrun (python -m torch.distributed.run)
        rank = int(os.getenv("RANK"))
        local_rank = int(os.getenv("LOCAL_RANK"))
        world_size = int(os.getenv("WORLD_SIZE"))
        local_size = int(os.getenv("LOCAL_WORLD_SIZE"))
    elif os.environ.get("OMPI_COMMAND"):
        # launched with mpirun
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        local_size = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])
    elif os.environ.get("SLURM_PROCID"):
        # launched with srun (SLURM)
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NPROCS"])
        local_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
    else:
        rank = 0
        local_rank = rank
        world_size = 1
        local_size = 1

    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    return rank, local_rank, world_size, local_size, num_workers


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process(rank: Optional[int] = None) -> bool:
    if rank is not None:
        return rank == 0
    return get_rank() == 0


def init_distributed(backend: str = "nccl", init_method: str = "env://", rank: Optional[int] = None,
                     world_size: Optional[int] = None) -> None:
    if not dist.is_available():
        raise RuntimeError("torch.distributed not available")
    if is_distributed():
        return
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )
    dist.barrier()


def cleanup_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def reduce_mean(tensor):
    """All-reduce a scalar tensor and return the mean across ranks."""
    if not is_distributed():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor
