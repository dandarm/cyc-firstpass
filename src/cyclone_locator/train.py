import os, argparse, yaml, time, csv, random
import numpy as np
import torch, torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from cyclone_locator.datasets.med_fullbasin import MedFullBasinDataset
from cyclone_locator.models.simplebaseline import SimpleBaseline
from cyclone_locator.losses.heatmap_loss import HeatmapMSE
from cyclone_locator.utils.distributed import (
    cleanup_distributed,
    get_resources,
    is_main_process,
    reduce_mean,
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yml")
    ap.add_argument("--train_csv")
    ap.add_argument("--val_csv")
    ap.add_argument("--test_csv")
    ap.add_argument("--image_size", type=int)
    ap.add_argument("--heatmap_stride", type=int)
    ap.add_argument("--heatmap_sigma_px", type=float)
    ap.add_argument("--backbone")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--bs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--log_dir")
    ap.add_argument("--best_ckpt_start_epoch", type=int)
    ap.add_argument("--num_workers", type=int, help="Override dataloader workers (use 0 to debug NCCL stalls)")
    ap.add_argument("--dataloader_timeout_s", type=int, help="Timeout (s) for DataLoader get_batch to avoid deadlocks")
    ap.add_argument("--persistent_workers", type=int, choices=[0,1], help="Set persistent_workers (1=keep workers alive between epochs)")
    return ap.parse_args()

def set_seed(sd):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sd)


def seed_worker(worker_id):
    """Ensure dataloader workers get different seeds."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def bce_logits(pred, target):
    return nn.functional.binary_cross_entropy_with_logits(pred, target)

def combine_losses(L_hm, L_pr, loss_cfg):
    hm_w = loss_cfg["w_heatmap"]
    pr_w = loss_cfg["w_presence"]
    return hm_w, pr_w, hm_w * L_hm + pr_w * L_pr

def evaluate_loader(model, loader, hm_loss, amp_enabled, loss_weights, device, rank: int = 0):
    vL, vHm, vPr = [], [], []
    bad_batches = 0
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            hm_t = batch["heatmap"].to(device, non_blocking=True)
            pres = batch["presence"].to(device, non_blocking=True)
            with autocast(enabled=amp_enabled):
                hm_p, pres_logit = model(img)
                L_hm = hm_loss(hm_p, hm_t)
                L_pr = bce_logits(pres_logit, pres)
                _, _, L = combine_losses(L_hm, L_pr, loss_weights)
            if not torch.isfinite(L):
                bad_batches += 1
                if bad_batches <= 3 and rank == 0:
                    print(f"[WARN] non-finite val loss (rank {rank}), skipping batch")
                continue
            vL.append(L.item()); vHm.append(L_hm.item()); vPr.append(L_pr.item())
    return {
        "loss": float(np.mean(vL)) if len(vL) > 0 else float("nan"),
        "hm": float(np.mean(vHm)) if len(vHm) > 0 else float("nan"),
        "presence": float(np.mean(vPr)) if len(vPr) > 0 else float("nan"),
        "bad_batches": bad_batches
    }

def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    # Optional CLI overrides keep backwards-compat with config-driven flow.
    if args.train_csv:
        cfg["data"]["manifest_train"] = args.train_csv
    if args.val_csv:
        cfg["data"]["manifest_val"] = args.val_csv
    if args.test_csv:
        cfg["data"]["manifest_test"] = args.test_csv
    if args.image_size:
        cfg["train"]["image_size"] = args.image_size
    if args.heatmap_stride:
        cfg["train"]["heatmap_stride"] = args.heatmap_stride
    if args.heatmap_sigma_px:
        cfg["loss"]["heatmap_sigma_px"] = args.heatmap_sigma_px
    if args.backbone:
        cfg["train"]["backbone"] = args.backbone
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.bs:
        cfg["train"]["batch_size"] = args.bs
    if args.lr:
        cfg["train"]["lr"] = args.lr
    if args.log_dir:
        cfg["train"]["save_dir"] = args.log_dir
    if args.best_ckpt_start_epoch is not None:
        cfg["train"]["best_ckpt_start_epoch"] = args.best_ckpt_start_epoch
    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers
    if args.dataloader_timeout_s is not None:
        cfg["train"]["dataloader_timeout_s"] = args.dataloader_timeout_s
    if args.persistent_workers is not None:
        cfg["train"]["persistent_workers"] = bool(args.persistent_workers)

    rank, local_rank, world_size, _, env_num_workers = get_resources()
    distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if distributed and device.type == "cuda":
        torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            init_method="env://",
            rank=rank,
            world_size=world_size
        )
        dist.barrier()

    set_seed(cfg["train"]["seed"] + rank)

    configured_workers = cfg["train"].get("num_workers", env_num_workers)
    num_workers = configured_workers if configured_workers is not None else env_num_workers
    if env_num_workers and num_workers is not None:
        num_workers = min(num_workers, env_num_workers)
    num_workers = int(num_workers or 0)
    cfg["train"]["num_workers"] = num_workers
    dl_timeout = int(cfg["train"].get("dataloader_timeout_s", 0) or 0)
    # PyTorch requires timeout=0 when num_workers=0 (single-process data loading).
    loader_timeout = dl_timeout if num_workers > 0 else 0
    persistent_workers_cfg = bool(cfg["train"].get("persistent_workers", True))
    persistent_workers = num_workers > 0 and persistent_workers_cfg
    pin_memory = device.type == "cuda"

    # Datasets
    ds_tr = MedFullBasinDataset(
        cfg["data"]["manifest_train"],
        image_size=cfg["train"]["image_size"],
        heatmap_stride=cfg["train"]["heatmap_stride"],
        heatmap_sigma_px=cfg["loss"]["heatmap_sigma_px"],
        use_aug=cfg["train"]["use_aug"],
        use_pre_letterboxed=cfg["data"]["use_pre_letterboxed"],
        letterbox_meta_csv=cfg["data"]["letterbox_meta_csv"],
        letterbox_size_assert=cfg["data"]["letterbox_size_assert"]
    )
    ds_va = MedFullBasinDataset(
        cfg["data"]["manifest_val"],
        image_size=cfg["train"]["image_size"],
        heatmap_stride=cfg["train"]["heatmap_stride"],
        heatmap_sigma_px=cfg["loss"]["heatmap_sigma_px"],
        use_aug=False,
        use_pre_letterboxed=cfg["data"]["use_pre_letterboxed"],
        letterbox_meta_csv=cfg["data"]["letterbox_meta_csv"],
        letterbox_size_assert=cfg["data"]["letterbox_size_assert"]
    )

    test_loader = None
    manifest_test = cfg["data"].get("manifest_test")
    if manifest_test:
        if os.path.exists(manifest_test):
            ds_te = MedFullBasinDataset(
                manifest_test,
                image_size=cfg["train"]["image_size"],
                heatmap_stride=cfg["train"]["heatmap_stride"],
                heatmap_sigma_px=cfg["loss"]["heatmap_sigma_px"],
                use_aug=False,
                use_pre_letterboxed=cfg["data"]["use_pre_letterboxed"],
                letterbox_meta_csv=cfg["data"]["letterbox_meta_csv"],
                letterbox_size_assert=cfg["data"]["letterbox_size_assert"]
            )
            test_loader = DataLoader(
                ds_te,
                batch_size=cfg["train"]["batch_size"],
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                worker_init_fn=seed_worker if num_workers > 0 else None
            )
        elif is_main_process(rank):
            print(f"[WARN] Test manifest {manifest_test} non trovato: salto l'eval di test.")

    train_sampler = DistributedSampler(
        ds_tr,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    ) if distributed else None

    tr_loader = DataLoader(
        ds_tr,
        batch_size=cfg["train"]["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        timeout=loader_timeout,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        drop_last=True
    )
    va_loader = DataLoader(
        ds_va,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        timeout=loader_timeout,
        worker_init_fn=seed_worker if num_workers > 0 else None
    )

    # Model
    model = SimpleBaseline(backbone=cfg["train"]["backbone"], out_heatmap_ch=1)
    model = model.to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None
        )
    model_to_save = model.module if distributed else model

    # Optim
    base_lr = cfg["train"]["lr"]
    scaled_lr = base_lr * world_size if (distributed and cfg["train"].get("scale_lr_by_world_size", True)) else base_lr
    opt = torch.optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=cfg["train"]["weight_decay"])
    scaler = GradScaler(enabled=cfg["train"]["amp"] and device.type == "cuda")
    hm_loss = HeatmapMSE()
    loss_weights = cfg["loss"]

    if is_main_process(rank):
        save_dir = cfg["train"]["save_dir"]; os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = cfg["train"]["save_dir"]
    if distributed:
        dist.barrier()

    log_path = os.path.join(save_dir, "training_log.csv")
    log_fields = [
        "epoch",
        "train_loss", "train_heatmap_loss", "train_presence_loss",
        "val_loss", "val_heatmap_loss", "val_presence_loss",
        "test_loss", "test_heatmap_loss", "test_presence_loss"
    ]
    if is_main_process(rank):
        print(f"[INFO] rank {rank}/{world_size} device={device}")
        print(f"[INFO] batch_size per GPU={cfg['train']['batch_size']} (global={cfg['train']['batch_size'] * world_size})")
        if scaled_lr != base_lr:
            print(f"[INFO] lr scaled from {base_lr} to {scaled_lr} for world_size={world_size}")
        print(f"Logging metrics to {log_path}")

    best_val = 1e9; best_path = None
    best_start_epoch = cfg["train"].get("best_ckpt_start_epoch", 1)

    log_file = None
    writer = None
    if is_main_process(rank):
        log_file = open(log_path, "w", newline="")
        writer = csv.DictWriter(log_file, fieldnames=log_fields)
        writer.writeheader()

    try:
        for epoch in range(1, cfg["train"]["epochs"]+1):
            epoch_start = time.time()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            losses = []; hm_losses = []; pres_losses = []; peak_preds = []
            for batch in tr_loader:
                img = batch["image"].to(device, non_blocking=True)
                hm_t = batch["heatmap"].to(device, non_blocking=True)
                pres = batch["presence"].to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with autocast(enabled=cfg["train"]["amp"]):
                    hm_p, pres_logit = model(img)
                    L_hm = hm_loss(hm_p, hm_t)
                    L_pr = bce_logits(pres_logit, pres)
                    _, _, L = combine_losses(L_hm, L_pr, loss_weights)
                if not torch.isfinite(L):
                    if is_main_process(rank):
                        print(f"[ERROR] non-finite train loss at epoch {epoch}; aborting to avoid hang.")
                    raise RuntimeError("Non-finite training loss")
                scaler.scale(L).backward()
                scaler.step(opt); scaler.update()

                losses.append(L.item()); hm_losses.append(L_hm.item()); pres_losses.append(L_pr.item())
                peak_preds.append(hm_p.detach().amax(dim=[-1, -2]).mean().item())

            tr_loss = float(np.mean(losses)) if losses else 0.0
            tr_hm = float(np.mean(hm_losses)) if hm_losses else 0.0
            tr_pr = float(np.mean(pres_losses)) if pres_losses else 0.0
            tr_peak = float(np.mean(peak_preds)) if peak_preds else 0.0

            metrics_tensor = torch.tensor([tr_loss, tr_hm, tr_pr, tr_peak], device=device)
            metrics_tensor = reduce_mean(metrics_tensor)
            tr_loss, tr_hm, tr_pr, tr_peak = [float(x) for x in metrics_tensor.tolist()]

            if is_main_process(rank):
                print(f"[Epoch {epoch}] train: L={tr_loss:.4f} (hm={tr_hm:.4f}, pr={tr_pr:.4f}, peak={tr_peak:.4f})")

            val_metrics = None; test_metrics = None
            if epoch % cfg["train"]["val_every"] == 0 and is_main_process(rank):
                model.eval()
                val_metrics = evaluate_loader(model, va_loader, hm_loss, cfg["train"]["amp"], loss_weights, device)
                if test_loader is not None:
                    test_metrics = evaluate_loader(model, test_loader, hm_loss, cfg["train"]["amp"], loss_weights, device)
                model.train()

                val_score = val_metrics["loss"]
                print(f"          val:  L={val_score:.4f} (hm={val_metrics['hm']:.4f}, pr={val_metrics['presence']:.4f})")

                if epoch < best_start_epoch:
                    print(f"[Epoch {epoch}] checkpoint skip: epoch < best_ckpt_start_epoch ({best_start_epoch})")
                elif val_score < best_val:
                    best_val = val_score
                    best_target = os.path.join(save_dir, "best.ckpt")
                    t0 = time.time()
                    print(f"[Epoch {epoch}] saving BEST checkpoint to {best_target} ...")
                    torch.save({"model": model_to_save.state_dict(), "cfg": cfg}, best_target)
                    print(f"[Epoch {epoch}] best checkpoint saved in {time.time()-t0:.2f}s")
                    best_path = best_target
                else:
                    print(f"[Epoch {epoch}] no val improvement ({val_score:.4f} >= {best_val:.4f}); checkpoint not saved")

            if writer is not None:
                row = {
                    "epoch": epoch,
                    "train_loss": tr_loss,
                    "train_heatmap_loss": tr_hm,
                    "train_presence_loss": tr_pr,
                    "val_loss": val_metrics["loss"] if val_metrics else "",
                    "val_heatmap_loss": val_metrics["hm"] if val_metrics else "",
                    "val_presence_loss": val_metrics["presence"] if val_metrics else "",
                    "test_loss": test_metrics["loss"] if test_metrics else "",
                    "test_heatmap_loss": test_metrics["hm"] if test_metrics else "",
                    "test_presence_loss": test_metrics["presence"] if test_metrics else ""
                }
                writer.writerow(row); log_file.flush()

            if distributed:
                dist.barrier()
            elapsed = time.time() - epoch_start
            if is_main_process(rank):
                print(f"[Epoch {epoch}] elapsed: {elapsed:.1f}s")
    finally:
        if log_file is not None:
            log_file.close()

    if is_main_process(rank):
        print("Done. Best:", best_path)
    cleanup_distributed()

if __name__ == "__main__":
    main()
