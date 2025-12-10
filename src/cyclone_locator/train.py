import os, argparse, yaml, time, csv
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from cyclone_locator.datasets.med_fullbasin import MedFullBasinDataset
from cyclone_locator.models.simplebaseline import SimpleBaseline
from cyclone_locator.losses.heatmap_loss import HeatmapMSE

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
    return ap.parse_args()

def set_seed(sd):
    import random, numpy as np, torch
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)

def bce_logits(pred, target):
    return nn.functional.binary_cross_entropy_with_logits(pred, target)

def combine_losses(L_hm, L_pr, loss_cfg):
    hm_w = loss_cfg["w_heatmap"]
    pr_w = loss_cfg["w_presence"]
    return hm_w, pr_w, hm_w * L_hm + pr_w * L_pr

def evaluate_loader(model, loader, hm_loss, amp_enabled, loss_weights):
    vL, vHm, vPr = [], [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].cuda(non_blocking=True)
            hm_t = batch["heatmap"].cuda(non_blocking=True)
            pres = batch["presence"].cuda(non_blocking=True)
            with autocast(enabled=amp_enabled):
                hm_p, pres_logit = model(img)
                L_hm = hm_loss(hm_p, hm_t)
                L_pr = bce_logits(pres_logit, pres)
                _, _, L = combine_losses(L_hm, L_pr, loss_weights)
            vL.append(L.item()); vHm.append(L_hm.item()); vPr.append(L_pr.item())
    return {
        "loss": float(np.mean(vL)),
        "hm": float(np.mean(vHm)),
        "presence": float(np.mean(vPr))
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

    set_seed(cfg["train"]["seed"])

    temporal_T = max(1, int(cfg["train"].get("temporal_T", 1)))
    temporal_stride = max(1, int(cfg["train"].get("temporal_stride", 1)))
    cfg["train"]["temporal_T"] = temporal_T
    cfg["train"]["temporal_stride"] = temporal_stride
    print(f"Temporal window: T={temporal_T}, stride={temporal_stride}")

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
            test_loader = DataLoader(
                ds_te,
                batch_size=cfg["train"]["batch_size"],
                shuffle=False,
                num_workers=cfg["train"]["num_workers"],
                pin_memory=True,
                persistent_workers=cfg["train"]["num_workers"] > 0
            )
        else:
            print(f"[WARN] Test manifest {manifest_test} non trovato: salto l'eval di test.")

    tr_loader = DataLoader(
        ds_tr,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["train"]["num_workers"] > 0,
        drop_last=True
    )
    va_loader = DataLoader(
        ds_va,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["train"]["num_workers"] > 0
    )

    # Model
    model = SimpleBaseline(
        backbone=cfg["train"]["backbone"],
        out_heatmap_ch=1,
        temporal_T=temporal_T
    )
    model = model.cuda()

    # Optim
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = GradScaler(enabled=cfg["train"]["amp"])
    hm_loss = HeatmapMSE()
    loss_weights = cfg["loss"]

    save_dir = cfg["train"]["save_dir"]; os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "training_log.csv")
    log_fields = [
        "epoch",
        "train_loss", "train_heatmap_loss", "train_presence_loss",
        "val_loss", "val_heatmap_loss", "val_presence_loss",
        "test_loss", "test_heatmap_loss", "test_presence_loss"
    ]
    print(f"Logging metrics to {log_path}")
    best_val = 1e9; best_path = None
    best_start_epoch = cfg["train"].get("best_ckpt_start_epoch", 1)

    with open(log_path, "w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=log_fields)
        writer.writeheader()

        for epoch in range(1, cfg["train"]["epochs"]+1):
            epoch_start = time.time()
            model.train()
            losses = []; hm_losses = []; pres_losses = []; peak_preds = []
            for batch in tr_loader:
                img = batch["image"].cuda(non_blocking=True)
                hm_t = batch["heatmap"].cuda(non_blocking=True)
                pres = batch["presence"].cuda(non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with autocast(enabled=cfg["train"]["amp"]):
                    hm_p, pres_logit = model(img)
                    L_hm = hm_loss(hm_p, hm_t)
                    L_pr = bce_logits(pres_logit, pres)
                    _, _, L = combine_losses(L_hm, L_pr, loss_weights)
                scaler.scale(L).backward()
                scaler.step(opt); scaler.update()

                losses.append(L.item()); hm_losses.append(L_hm.item()); pres_losses.append(L_pr.item())
                peak_preds.append(hm_p.detach().amax(dim=[-1, -2]).mean().item())

            tr_loss = float(np.mean(losses))
            tr_hm = float(np.mean(hm_losses))
            tr_pr = float(np.mean(pres_losses))
            tr_peak = float(np.mean(peak_preds))
            print(f"[Epoch {epoch}] train: L={tr_loss:.4f} (hm={tr_hm:.4f}, pr={tr_pr:.4f}, peak={tr_peak:.4f})")

            val_metrics = None; test_metrics = None
            if epoch % cfg["train"]["val_every"] == 0:
                model.eval()
                val_metrics = evaluate_loader(model, va_loader, hm_loss, cfg["train"]["amp"], loss_weights)
                if test_loader is not None:
                    test_metrics = evaluate_loader(model, test_loader, hm_loss, cfg["train"]["amp"], loss_weights)
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
                    torch.save({"model": model.state_dict(), "cfg": cfg}, best_target)
                    print(f"[Epoch {epoch}] best checkpoint saved in {time.time()-t0:.2f}s")
                    best_path = best_target
                else:
                    print(f"[Epoch {epoch}] no val improvement ({val_score:.4f} >= {best_val:.4f}); checkpoint not saved")

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
            elapsed = time.time() - epoch_start
            print(f"[Epoch {epoch}] elapsed: {elapsed:.1f}s")

    print("Done. Best:", best_path)

if __name__ == "__main__":
    main()
