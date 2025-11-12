#!/usr/bin/env python3
"""Generate train/val/test manifests from temporal windows CSV."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from cyclone_locator.datasets.windows_labeling import (
    WindowsLabeling,
    compute_letterbox_params,
    parse_timestamp_from_filename,
    project_keypoint,
)


def list_images(images_dir: Path, exts: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for ext in exts:
        files.extend(sorted(images_dir.glob(f"**/*{ext}")))
    return sorted(set(files))


def stratified_split_indices(labels: Sequence[int], val_frac: float, test_frac: float, seed: int) -> Dict[str, List[int]]:
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be >=0 and sum to less than 1")
    rng = np.random.default_rng(seed)
    grouped: Dict[int, List[int]] = defaultdict(list)
    for idx, lbl in enumerate(labels):
        grouped[int(lbl)].append(idx)

    splits = {"train": [], "val": [], "test": []}
    for idxs in grouped.values():
        arr = np.array(idxs, dtype=int)
        rng.shuffle(arr)
        n = len(arr)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        if n_test + n_val > n:
            overflow = n_test + n_val - n
            if n_test >= overflow:
                n_test -= overflow
            else:
                overflow -= n_test
                n_test = 0
                n_val = max(0, n_val - overflow)
        test_idx = arr[:n_test]
        val_idx = arr[n_test:n_test + n_val]
        train_idx = arr[n_test + n_val:]
        splits["test"].extend(test_idx.tolist())
        splits["val"].extend(val_idx.tolist())
        splits["train"].extend(train_idx.tolist())
    return splits


def build_master_dataframe(
    image_paths: Sequence[Path],
    labeler: WindowsLabeling,
    attach_keypoints: bool,
    letterbox_params,
) -> tuple[pd.DataFrame, int]:
    records: List[Dict[str, object]] = []
    missing_ts: List[Path] = []
    keypoint_attachments = 0
    for path in sorted(image_paths):
        timestamp = parse_timestamp_from_filename(path)
        if timestamp is None:
            missing_ts.append(path)
            continue
        presence = 1 if labeler.is_positive(timestamp) else 0
        record: Dict[str, object] = {
            "image_path": str(path.resolve()),
            "datetime": timestamp,
            "presence": presence,
        }
        kp = labeler.keypoint_for(timestamp)
        if presence == 1 and kp is not None:
            record["cx"] = kp.x
            record["cy"] = kp.y
            if attach_keypoints:
                x_lb, y_lb = project_keypoint(kp.x, kp.y, letterbox_params)
                record["x_pix_resized"] = x_lb
                record["y_pix_resized"] = y_lb
                keypoint_attachments += 1
        else:
            record["cx"] = np.nan
            record["cy"] = np.nan
            if attach_keypoints:
                record["x_pix_resized"] = np.nan
                record["y_pix_resized"] = np.nan
        records.append(record)

    if missing_ts:
        raise RuntimeError(
            f"Could not parse timestamp from {len(missing_ts)} files, e.g. {missing_ts[0]}"
        )

    df = pd.DataFrame.from_records(records)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    if attach_keypoints:
        df["x_pix_resized"] = df["x_pix_resized"].astype("float")
        df["y_pix_resized"] = df["y_pix_resized"].astype("float")
    return df, keypoint_attachments


def save_manifest(df: pd.DataFrame, path: Path) -> None:
    df_to_save = df.copy()
    df_to_save["datetime"] = df_to_save["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_to_save.to_csv(path, index=False)


def write_log(out_dir: Path, summary: Dict[str, object]) -> None:
    log_path = out_dir / "log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate manifests from temporal windows")
    ap.add_argument("--windows-csv", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--orig-size", type=int, nargs=2, default=[1290, 420], metavar=("W", "H"))
    ap.add_argument("--target-size", type=int, default=512)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--test-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attach-keypoints", choices=["auto", "never"], default="auto")
    ap.add_argument("--exts", default=".png,.jpg,.jpeg", help="comma separated image extensions")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows_csv = Path(args.windows_csv)
    images_dir = Path(args.images_dir)
    if not windows_csv.exists():
        raise FileNotFoundError(windows_csv)
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)

    labeler = WindowsLabeling.from_csv(windows_csv)
    attach_keypoints = False
    if args.attach_keypoints == "auto":
        attach_keypoints = labeler.has_keypoints()
    letterbox_params = compute_letterbox_params(args.orig_size[0], args.orig_size[1], args.target_size)

    exts = [ext.strip() for ext in args.exts.split(",") if ext.strip()]
    image_paths = list_images(images_dir, exts)
    if not image_paths:
        raise RuntimeError("No images found in images-dir")

    master_df, keypoint_attachments = build_master_dataframe(
        image_paths,
        labeler,
        attach_keypoints,
        letterbox_params,
    )

    labels = master_df["presence"].astype(int).tolist()
    splits = stratified_split_indices(labels, args.val_split, args.test_split, args.seed)

    manifests = {}
    for split_name, indices in splits.items():
        df_split = master_df.iloc[indices].copy()
        df_split.sort_values("datetime", inplace=True)
        manifests[split_name] = df_split

    save_manifest(manifests["train"], out_dir / "train.csv")
    save_manifest(manifests["val"], out_dir / "val.csv")
    save_manifest(manifests["test"], out_dir / "test.csv")

    total_frames = len(master_df)
    total_positive = int(master_df["presence"].sum())
    positive_pct = (total_positive / total_frames) * 100.0
    keypoint_pct = (keypoint_attachments / total_frames) * 100.0 if total_frames else 0.0
    summary = {
        "frames_total": total_frames,
        "frames_positive": total_positive,
        "positive_pct": f"{positive_pct:.2f}%",
        "keypoints_attached": keypoint_attachments,
        "keypoints_pct": f"{keypoint_pct:.2f}%",
        "train_count": len(manifests["train"]),
        "val_count": len(manifests["val"]),
        "test_count": len(manifests["test"]),
    }
    write_log(out_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
