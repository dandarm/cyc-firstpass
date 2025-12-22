#!/usr/bin/env python
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Debug peak/logit stats for a single sample")
    ap.add_argument("--config", default="config/default.yml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--index", type=int, default=None, help="Dataset index (0-based). If omitted, picks first negative.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from cyclone_locator.datasets.med_fullbasin import MedFullBasinDataset
    from cyclone_locator.models.simplebaseline import SimpleBaseline

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("cfg")
    if cfg is None:
        cfg = yaml.safe_load(open(repo_root / args.config))

    split_key = {"train": "manifest_train", "val": "manifest_val", "test": "manifest_test"}[args.split]
    manifest = cfg["data"][split_key]
    backbone = cfg["train"]["backbone"]
    pretrained = bool(cfg["train"].get("backbone_pretrained", True))
    temporal_T = int(cfg["train"].get("temporal_T", 1))
    temporal_stride = int(cfg["train"].get("temporal_stride", 1))

    if str(backbone).startswith("x3d"):
        try:
            from cyclone_locator.models.x3d_backbone import X3DBackbone
        except Exception as e:
            raise SystemExit(
                f"Cannot import X3DBackbone ({e}). "
                "On local dev you may be missing pytorchvideo; run this script on the training environment."
            )
        model = X3DBackbone(
            backbone=backbone,
            out_heatmap_ch=1,
            presence_dropout=cfg["train"].get("presence_dropout", 0.0),
            pretrained=pretrained,
        )
    else:
        model = SimpleBaseline(
            backbone=backbone,
            out_heatmap_ch=1,
            temporal_T=temporal_T,
            presence_dropout=cfg["train"].get("presence_dropout", 0.0),
            pretrained=pretrained,
        )

    weights = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    print(f"backbone={backbone} pretrained={pretrained} missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()

    ds = MedFullBasinDataset(
        csv_path=str(repo_root / manifest),
        image_size=cfg["train"]["image_size"],
        heatmap_stride=cfg["train"]["heatmap_stride"],
        heatmap_sigma_px=cfg["loss"]["heatmap_sigma_px"],
        use_aug=False,
        use_pre_letterboxed=cfg["data"].get("use_pre_letterboxed", True),
        letterbox_meta_csv=str(repo_root / cfg["data"]["letterbox_meta_csv"]),
        letterbox_size_assert=cfg["data"].get("letterbox_size_assert"),
        temporal_T=temporal_T,
        temporal_stride=temporal_stride,
    )

    if args.index is None:
        neg = np.where(ds.df["presence"].to_numpy() == 0)[0]
        if len(neg) == 0:
            raise SystemExit("No negative samples found in this split.")
        idx = int(neg[0])
    else:
        idx = int(args.index)

    sample = ds[idx]
    input_key = "video" if getattr(model, "input_is_video", False) else "image"
    x = sample[input_key].unsqueeze(0)

    with torch.no_grad():
        hm_logits, pres_logit = model(x)

    hm_logits = hm_logits.squeeze(0).squeeze(0)  # (H,W)
    peak_logit = float(hm_logits.max().item())
    peak_sigmoid = float(torch.sigmoid(torch.tensor(peak_logit)).item())
    print(f"idx={idx} presence_gt={float(sample['presence'].item()):.3f} path={sample.get('image_path')}")
    print(f"hm_logits min={float(hm_logits.min()):.6f} max={float(hm_logits.max()):.6f}")
    print(f"peak_logit={peak_logit:.6f} sigmoid(peak_logit)={peak_sigmoid:.6f}")
    print(f"presence_logit_head={float(pres_logit.squeeze().item()):.6f} sigmoid={float(torch.sigmoid(pres_logit).squeeze().item()):.6f}")


if __name__ == "__main__":
    main()

