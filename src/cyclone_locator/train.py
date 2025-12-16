import os, argparse, yaml, time, csv, random
import numpy as np
import torch, torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from cyclone_locator.datasets.med_fullbasin import MedFullBasinDataset
from cyclone_locator.models.simplebaseline import SimpleBaseline
from cyclone_locator.models.x3d_backbone import X3DBackbone
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
    ap.add_argument("--temporal_T", type=int)
    ap.add_argument("--temporal_stride", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--bs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--log_dir")
    ap.add_argument("--best_ckpt_start_epoch", type=int)
    ap.add_argument("--num_workers", type=int, help="Override dataloader workers (use 0 to debug NCCL stalls)")
    ap.add_argument("--dataloader_timeout_s", type=int, help="Timeout (s) for DataLoader get_batch to avoid deadlocks")
    ap.add_argument("--persistent_workers", type=int, choices=[0,1], help="Set persistent_workers (1=keep workers alive between epochs)")
    ap.add_argument("--heatmap_neg_multiplier", type=float, help="Scale factor for heatmap loss on negative samples")
    ap.add_argument("--presence_from_peak", type=int, choices=[0,1], help="If 1, disable presence head and use heatmap peak as presence prob")
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

def smooth_targets(target, eps: float):
    if eps <= 0.0:
        return target
    return target * (1.0 - eps) + (1.0 - target) * eps

def combine_losses(L_hm, L_pr, loss_cfg):
    hm_w = loss_cfg["w_heatmap"]
    pr_w = loss_cfg["w_presence"]
    return hm_w, pr_w, hm_w * L_hm + pr_w * L_pr


def focal_loss_with_logits(logits, targets, gamma: float = 2.0, alpha: float = None):
    """
    Binary focal loss on logits. Targets are probabilities (after smoothing).
    """
    ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_factor = (1 - p_t).pow(gamma)
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        ce = ce * alpha_t
    return (focal_factor * ce).mean()


def combine_presence_probs(logits: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
    """Replica la logica di inferenza: media tra sigmoid(logit) e picco heatmap."""
    if heatmap.ndim == 4 and heatmap.shape[1] == 1:
        heatmap = heatmap.squeeze(1)
    probs = torch.sigmoid(logits).squeeze(1)
    peak = heatmap.amax(dim=[-1, -2])
    combined = 0.5 * probs + 0.5 * peak
    combined = torch.clamp(combined, 1e-6, 1 - 1e-6)  # per BCE numericamente stabile
    return combined

def log_batch_temporal_samples(batch, temporal_selector, tag="train", max_samples=3):
    T = temporal_selector.temporal_T
    if T <= 1:
        return
    paths = batch.get("image_path_abs") or batch.get("image_path")
    if paths is None:
        return
    if isinstance(paths, str):
        paths = [paths]
    count = min(max_samples, len(paths))
    print(f"[Debug] {tag}: showing {count} temporal samples (T={T}, stride={temporal_selector.temporal_stride})")
    for i in range(count):
        center_path = paths[i]
        window = temporal_selector.get_window(center_path)
        window_str = ", ".join(os.path.basename(p) for p in window)
        print(f"  sample {i}: {os.path.basename(center_path)} -> [{window_str}]")

def log_temporal_debug_samples(dataset, tag="train", max_samples=3):
    T = dataset.temporal_selector.temporal_T
    stride = dataset.temporal_selector.temporal_stride
    if T <= 1 or len(dataset) == 0:
        return
    step = max(1, len(dataset) // max_samples)
    indices = [min(i, len(dataset) - 1) for i in range(0, len(dataset), step)][:max_samples]
    print(f"[Debug] {tag} temporal samples (T={T}, stride={stride})")
    for idx in indices:
        row = dataset.df.iloc[idx]
        center_abs = row["image_path_abs"]
        window = dataset.temporal_selector.get_window(center_abs)
        window_str = ", ".join(os.path.basename(p) for p in window)
        print(f"  idx={idx}: {os.path.basename(row['image_path'])} -> [{window_str}]")


def evaluate_loader(model, loader, hm_loss, amp_enabled, loss_weights, device, rank: int = 0,
                    distributed: bool = False, world_size: int = 1,
                    presence_smoothing: float = 0.0,
                    presence_loss_fn=None,
                    log_combined_presence: bool = False,
                    input_key: str = "image",
                    presence_from_peak: bool = False):
    #vL, vHm, vPr = [], [], []
    sum_L, sum_hm, sum_pr, sum_pr_comb = 0.0, 0.0, 0.0, 0.0
    bad_batches = 0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            total_batches += 1
            img = batch[input_key].to(device, non_blocking=True)
            hm_t = batch["heatmap"].to(device, non_blocking=True)
            pres = batch["presence"].to(device, non_blocking=True)
            pres_smooth = smooth_targets(pres, presence_smoothing)
            # Val in full precision to reduce risk of overflow/NaN
            with autocast(enabled=amp_enabled):
                hm_p, pres_logit = model(img)
                hm_p = torch.nan_to_num(hm_p, nan=0.0, posinf=1e4, neginf=-1e4)
                pres_logit = torch.nan_to_num(pres_logit, nan=0.0, posinf=50.0, neginf=-50.0)
                hm_per = hm_loss(hm_p, hm_t, reduction="none")
                pos_mask = (pres.squeeze(1) > 0.5).float()
                neg_mask = 1.0 - pos_mask
                neg_mult = float(loss_weights.get("heatmap_neg_multiplier", 1.0) or 1.0)
                weight = pos_mask + neg_mult * neg_mask
                L_hm = (hm_per * weight).sum() / torch.clamp(weight.sum(), min=1.0)
                if presence_from_peak:
                    L_pr = torch.tensor(0.0, device=device, dtype=hm_p.dtype)
                    comb_prob = torch.clamp(hm_p.amax(dim=[-1, -2]), 1e-6, 1 - 1e-6)
                    L_pr_comb = nn.functional.binary_cross_entropy(comb_prob, pres_smooth.view_as(comb_prob)) if log_combined_presence else torch.tensor(0.0, device=device)
                else:
                    L_pr = presence_loss_fn(pres_logit, pres_smooth) if presence_loss_fn else bce_logits(pres_logit, pres_smooth)
                    if log_combined_presence:
                        comb_prob = combine_presence_probs(pres_logit, hm_p)
                        L_pr_comb = nn.functional.binary_cross_entropy(comb_prob, pres_smooth.view_as(comb_prob))
                    else:
                        L_pr_comb = torch.tensor(0.0, device=device)
                _, _, L = combine_losses(L_hm, L_pr, loss_weights)
            if not torch.isfinite(L):
                bad_batches += 1
                if bad_batches <= 3 and rank == 0:
                    print(f"[WARN] non-finite val loss (rank {rank}), skipping batch")
                continue
            sum_L += L.item(); sum_hm += L_hm.item(); sum_pr += L_pr.item(); sum_pr_comb += L_pr_comb.item()

    totals = torch.tensor(
        [sum_L, sum_hm, sum_pr, sum_pr_comb, float(total_batches), float(bad_batches)],
        device=device
    )
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    sum_L, sum_hm, sum_pr, sum_pr_comb, total_batches, bad_batches = totals.tolist()
    if total_batches - bad_batches > 0:
        mean_L = sum_L / (total_batches - bad_batches)
        mean_hm = sum_hm / (total_batches - bad_batches)
        mean_pr = sum_pr / (total_batches - bad_batches)
        mean_pr_comb = sum_pr_comb / (total_batches - bad_batches)
    else:
        mean_L = mean_hm = mean_pr = mean_pr_comb = float("nan")
    return {
        "loss": float(mean_L),
        "hm": float(mean_hm),
        "presence": float(mean_pr),
        "presence_combined": float(mean_pr_comb) if log_combined_presence else None,
        "bad_batches": int(bad_batches),
        "total_batches": int(total_batches)
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
    if args.heatmap_neg_multiplier is not None:
        cfg["loss"]["heatmap_neg_multiplier"] = args.heatmap_neg_multiplier
    if args.presence_from_peak is not None:
        cfg["train"]["presence_from_peak"] = bool(args.presence_from_peak)
    if args.backbone:
        cfg["train"]["backbone"] = args.backbone
    if args.temporal_T:
        cfg["train"]["temporal_T"] = args.temporal_T
    if args.temporal_stride:
        cfg["train"]["temporal_stride"] = args.temporal_stride
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

    # temporal parameters
    temporal_T = max(1, int(cfg["train"].get("temporal_T", 1)))
    temporal_stride = max(1, int(cfg["train"].get("temporal_stride", 1)))
    cfg["train"]["temporal_T"] = temporal_T
    cfg["train"]["temporal_stride"] = temporal_stride
    print(f"Temporal window: T={temporal_T}, stride={temporal_stride}")


    # Distributed setup
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
        letterbox_size_assert=cfg["data"]["letterbox_size_assert"],
        temporal_T=temporal_T,
        temporal_stride=temporal_stride
    )
    #log_temporal_debug_samples(ds_tr, tag="train")
    ds_va = MedFullBasinDataset(
        cfg["data"]["manifest_val"],
        image_size=cfg["train"]["image_size"],
        heatmap_stride=cfg["train"]["heatmap_stride"],
        heatmap_sigma_px=cfg["loss"]["heatmap_sigma_px"],
        use_aug=False,
        use_pre_letterboxed=cfg["data"]["use_pre_letterboxed"],
        letterbox_meta_csv=cfg["data"]["letterbox_meta_csv"],
        letterbox_size_assert=cfg["data"]["letterbox_size_assert"],
        temporal_T=temporal_T,
        temporal_stride=temporal_stride
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
                letterbox_size_assert=cfg["data"]["letterbox_size_assert"],
                temporal_T=temporal_T,
                temporal_stride=temporal_stride
            )

            test_sampler = DistributedSampler(
                ds_te,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False
            ) if distributed else None
            test_loader = DataLoader(
                ds_te,
                batch_size=cfg["train"]["batch_size"],
                shuffle=test_sampler is None,
                sampler=test_sampler,
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
    va_sampler = DistributedSampler(
        ds_va,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    ) if distributed else None

    va_loader = DataLoader(
        ds_va,
        batch_size=cfg["train"]["batch_size"],
        shuffle=va_sampler is None,
        sampler=va_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        timeout=loader_timeout,
        worker_init_fn=seed_worker if num_workers > 0 else None
    )

    presence_from_peak = bool(cfg["train"].get("presence_from_peak", False))

    # Model
    backbone_name = cfg["train"]["backbone"]
    pretrained_backbone = bool(cfg["train"].get("backbone_pretrained", True))
    if backbone_name.startswith("x3d"):
        model = X3DBackbone(
            backbone=backbone_name,
            out_heatmap_ch=1,
            presence_dropout=cfg["train"].get("presence_dropout", 0.0),
            pretrained=pretrained_backbone,
        )
    else:
        model = SimpleBaseline(
            backbone=backbone_name,
            out_heatmap_ch=1,
            temporal_T=temporal_T,
            presence_dropout=cfg["train"].get("presence_dropout", 0.0),
            pretrained=pretrained_backbone,
        )
    model = model.to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=presence_from_peak  # presence head non usata se si usa solo il picco
        )
    model_to_save = model.module if distributed else model
    input_key = "video" if getattr(model_to_save, "input_is_video", False) else "image"


    # Optim
    base_lr = cfg["train"]["lr"]
    scaled_lr = base_lr * world_size if (distributed and cfg["train"].get("scale_lr_by_world_size", True)) else base_lr
    opt = torch.optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=cfg["train"]["weight_decay"])
    scaler = GradScaler(enabled=cfg["train"]["amp"] and device.type == "cuda")
    hm_loss = HeatmapMSE()
    loss_weights = cfg["loss"]
    presence_smoothing = float(cfg["loss"].get("presence_label_smoothing", 0.0) or 0.0)
    presence_loss_type = cfg["loss"].get("presence_loss", "bce")
    focal_gamma = float(cfg["loss"].get("presence_focal_gamma", 2.0) or 2.0)
    focal_alpha = cfg["loss"].get("presence_focal_alpha", None)
    if focal_alpha is not None:
        focal_alpha = float(focal_alpha)
    if presence_loss_type == "focal":
        def presence_loss_fn(logits, target):
            return focal_loss_with_logits(logits, target, gamma=focal_gamma, alpha=focal_alpha)
    else:
        def presence_loss_fn(logits, target):
            return bce_logits(logits, target)

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
        "test_loss", "test_heatmap_loss", "test_presence_loss", "test_presence_combined_loss"
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
                #log_batch_temporal_samples(batch, ds_tr.temporal_selector, tag=f"train batch ", max_samples=3)   #{batch_idx}
                img = batch[input_key].to(device, non_blocking=True)
                hm_t = batch["heatmap"].to(device, non_blocking=True)
                pres = batch["presence"].to(device, non_blocking=True)
                pres_smooth = smooth_targets(pres, presence_smoothing)
                pres_raw = pres.squeeze(1)

                opt.zero_grad(set_to_none=True)
                with autocast(enabled=cfg["train"]["amp"]):
                    hm_p, pres_logit = model(img)
                    hm_per = hm_loss(hm_p, hm_t, reduction="none")
                    pos_mask = (pres_raw > 0.5).float()
                    neg_mask = 1.0 - pos_mask
                    neg_mult = float(cfg["loss"].get("heatmap_neg_multiplier", 1.0) or 1.0)
                    weight = pos_mask + neg_mult * neg_mask
                    L_hm = (hm_per * weight).sum() / torch.clamp(weight.sum(), min=1.0)
                    if presence_from_peak:
                        # salta la head presence: perdita nulla, presence logit non usato
                        L_pr = torch.tensor(0.0, device=device, dtype=L_hm.dtype)
                    else:
                        L_pr = presence_loss_fn(pres_logit, pres_smooth)
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
            if epoch % cfg["train"]["val_every"] == 0:
                if distributed:
                    dist.barrier()

                eval_model = model.module if distributed else model
                eval_model.eval()
                val_metrics = evaluate_loader(
                    eval_model, va_loader, hm_loss, False, loss_weights, device,
                    rank=rank, distributed=distributed, world_size=world_size,
                    presence_smoothing=presence_smoothing,
                    presence_loss_fn=presence_loss_fn,
                    log_combined_presence=False,
                    input_key=input_key,
                    presence_from_peak=presence_from_peak
                )
                if test_loader is not None:
                    test_metrics = evaluate_loader(
                        eval_model, test_loader, hm_loss, False, loss_weights, device,
                        rank=rank, distributed=distributed, world_size=world_size,
                        presence_smoothing=presence_smoothing,
                        presence_loss_fn=presence_loss_fn,
                        log_combined_presence=True,
                        input_key=input_key,
                        presence_from_peak=presence_from_peak
                    )
                eval_model.train()

                if distributed:
                    dist.barrier()

                if is_main_process(rank) and val_metrics is not None:
                    val_score = val_metrics["loss"]
                    if val_metrics.get("bad_batches", 0) > 0:
                        total = val_metrics.get("total_batches", 0)
                        msg = f"[WARN] val skipped {val_metrics['bad_batches']} non-finite batches"
                        if total:
                            msg += f" out of {total}"
                        print(msg)
                        if total and val_metrics["bad_batches"] == total:
                            raise RuntimeError("Validation produced only non-finite batches; aborting.")
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
                    "test_presence_loss": test_metrics["presence"] if test_metrics else "",
                    "test_presence_combined_loss": test_metrics["presence_combined"] if test_metrics else ""
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
