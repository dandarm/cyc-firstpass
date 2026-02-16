#!/usr/bin/env python3
"""Generate train/val/test manifests from temporal windows CSV."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from medicane_utils.buil_dataset import get_train_test_validation_df

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


def build_master_dataframe(
    image_paths: Sequence[Path],
    labeler: WindowsLabeling,
    attach_keypoints: bool,
    project_fn: Optional[Callable[[float, float], tuple[int, int]]],
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
            "event_id": labeler.event_id_for(timestamp),
        }
        kp = labeler.keypoint_for(timestamp)
        if presence == 1 and kp is not None:
            record["cx"] = kp.x
            record["cy"] = kp.y
            if attach_keypoints and project_fn is not None:
                x_lb, y_lb = project_fn(kp.x, kp.y)
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


def project_keypoint_stretch(
    x: float,
    y: float,
    src_w: int,
    src_h: int,
    target_size: int,
) -> tuple[int, int]:
    sx = target_size / float(src_w)
    sy = target_size / float(src_h)
    x_lb = int(round(x * sx))
    y_lb = int(round(y * sy))
    clipped_x = int(np.clip(x_lb, 0, target_size - 1))
    clipped_y = int(np.clip(y_lb, 0, target_size - 1))
    return clipped_x, clipped_y


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
    ap.add_argument(
        "--resize-mode",
        choices=["letterbox", "stretch"],
        default="letterbox",
        help="Modalità resize usata per le immagini (letterbox con padding oppure stretch).",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--test-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--buffer-hours",
        type=float,
        default=0.0,
        help="Conserva i frame (anche label 0) entro questa distanza oraria dai bordi delle finestre",
    )
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

    windows_df = pd.read_csv(windows_csv, parse_dates=["time", "start_time", "end_time"], keep_default_na=True)
    event_id_col = WindowsLabeling._detect_event_id_column(windows_df)  # type: ignore[attr-defined]
    if event_id_col is None:
        raise ValueError("Non trovo una colonna id evento (es. id_final) in windows_csv")

    train_pct = 1.0 - args.val_split - args.test_split
    if train_pct <= 0:
        raise ValueError("val_split + test_split devono essere < 1")

    df_train, df_test, df_val = get_train_test_validation_df(
        windows_df,
        percentage=train_pct,
        validation_percentage=args.val_split,
        id_col=event_id_col,
        verbose=False,
    )
    normalize_id = WindowsLabeling._normalize_event_id  # type: ignore[attr-defined]
    train_ids = {normalize_id(v) for v in df_train[event_id_col].unique() if pd.notna(v)}
    val_ids = {normalize_id(v) for v in df_val[event_id_col].unique() if pd.notna(v)}
    test_ids = {normalize_id(v) for v in df_test[event_id_col].unique() if pd.notna(v)}
    for id_set in (train_ids, val_ids, test_ids):
        id_set.discard(None)
    event_split = {eid: "train" for eid in train_ids}
    event_split.update({eid: "val" for eid in val_ids})
    event_split.update({eid: "test" for eid in test_ids})

    labeler = WindowsLabeling.from_dataframe(windows_df)
    attach_keypoints = False
    if args.attach_keypoints == "auto":
        attach_keypoints = labeler.has_keypoints()
    project_fn = None
    if attach_keypoints:
        if args.resize_mode == "letterbox":
            letterbox_params = compute_letterbox_params(args.orig_size[0], args.orig_size[1], args.target_size)
            project_fn = lambda x, y: project_keypoint(x, y, letterbox_params)
        else:
            project_fn = lambda x, y: project_keypoint_stretch(
                x,
                y,
                args.orig_size[0],
                args.orig_size[1],
                args.target_size,
            )

    exts = [ext.strip() for ext in args.exts.split(",") if ext.strip()]
    image_paths = list_images(images_dir, exts)
    if not image_paths:
        raise RuntimeError("No images found in images-dir")

    master_df, keypoint_attachments = build_master_dataframe(
        image_paths,
        labeler,
        attach_keypoints,
        project_fn,
    )

    if args.buffer_hours > 0:
        buffer = pd.Timedelta(hours=args.buffer_hours)
        intervals = labeler.intervals

        def in_buffer(ts: pd.Timestamp) -> bool:
            return any((ts >= start - buffer) and (ts <= end + buffer) for start, end in intervals)

        mask_pos = master_df["presence"] == 1
        mask_buffer = master_df["datetime"].apply(in_buffer)
        master_df = master_df[mask_pos | mask_buffer].reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    master_df["split"] = None
    event_ids = master_df["event_id"].apply(normalize_id)
    master_df.loc[event_ids.isin(train_ids), "split"] = "train"
    master_df.loc[event_ids.isin(val_ids), "split"] = "val"
    master_df.loc[event_ids.isin(test_ids), "split"] = "test"

    # distribuisci background senza event_id in modo riproducibile secondo le frazioni richieste
    mask_unassigned = master_df["split"].isna()
    if mask_unassigned.any():
        max_gap = pd.Timedelta(days=7)
        reassigned = 0
        for idx, row in master_df.loc[mask_unassigned].iterrows():
            eid, gap = labeler.nearest_event(row["datetime"])
            eid_norm = normalize_id(eid) if eid is not None else None
            if eid_norm and gap is not None and gap <= max_gap and eid_norm in event_split:
                master_df.at[idx, "split"] = event_split[eid_norm]
                master_df.at[idx, "event_id"] = eid_norm
                reassigned += 1
        if reassigned:
            print(f"[INFO] Assegnate {reassigned} righe senza event_id allo split dell'evento più vicino (<= {max_gap})")

    mask_unassigned = master_df["split"].isna()
    if mask_unassigned.any():
        probs = np.array([train_pct, args.val_split, args.test_split], dtype=float)
        probs = probs / probs.sum()
        choices = rng.choice(["train", "val", "test"], size=mask_unassigned.sum(), p=probs)
        master_df.loc[mask_unassigned, "split"] = choices
        print(f"[INFO] Assegnate {mask_unassigned.sum()} righe senza event_id con split casuale riproducibile")

    if master_df["split"].isna().any():
        raise RuntimeError("Alcune righe non hanno split assegnato")

    manifests = {}
    for split_name in ("train", "val", "test"):
        df_split = master_df[master_df["split"] == split_name].copy()
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
