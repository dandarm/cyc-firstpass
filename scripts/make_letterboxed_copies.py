#!/usr/bin/env python3
"""Seleziona frame in finestre temporali (con buffer), li letterboxa e salva letterbox_meta."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from cyclone_locator.transforms.letterbox import resize_image  # noqa: E402
from time_windows import (  # noqa: E402
    CSV_ID_CANDIDATES,
    DEFAULT_CSV_TIME_FORMATS,
    DEFAULT_EXTS,
    DEFAULT_FILENAME_FORMATS,
    DEFAULT_TIME_REGEX,
    load_windows,
    parse_dt_from_filename,
)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def iter_files(root: Path, recursive: bool, exts: Set[str]) -> Iterable[Path]:
    if recursive:
        return (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    return (p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Copy frames inside/around time windows, letterbox them, and write letterbox_meta + manifest.",
    )
    p.add_argument("--windows-csv", required=True, type=Path, help="CSV con start_time/end_time e id evento")
    p.add_argument("--src", required=True, type=Path, help="Directory di origine con i frame")
    p.add_argument("--out-dir", required=True, type=Path, help="Cartella di destinazione (conterrà resized/ e CSV)")
    p.add_argument("--image-size", type=int, default=512, help="Lato dell'immagine letterbox")
    p.add_argument(
        "--resize-mode",
        choices=["letterbox", "stretch"],
        default="letterbox",
        help="Modalità resize: letterbox (AR + padding) oppure stretch (distorsione senza padding).",
    )
    p.add_argument(
        "--buffer-hours",
        type=float,
        default=0.0,
        help="Margine prima/dopo ciascuna finestra, come in make_manifest_from_windows",
    )
    p.add_argument("--recurse", action="store_true", help="Scansiona ricorsivamente la sorgente")
    p.add_argument("--preserve-structure", action="store_true", help="Replica la struttura delle cartelle in resized/")
    p.add_argument("--time-regex", type=str, default=DEFAULT_TIME_REGEX, help="Regex per estrarre la data dal filename")
    p.add_argument(
        "--strptime",
        nargs="*",
        default=list(DEFAULT_FILENAME_FORMATS),
        help="Formati datetime nel filename, es. yyyy-MM-ddTHH-mm yyyyMMdd_HHmm",
    )
    p.add_argument(
        "--csv-strptime",
        nargs="*",
        default=list(DEFAULT_CSV_TIME_FORMATS),
        help="Formati datetime nel CSV se non ISO",
    )
    p.add_argument("--id-cols", nargs="*", default=list(CSV_ID_CANDIDATES), help="Colonne candidate per event_id nel CSV")
    p.add_argument("--ext", nargs="*", default=list(DEFAULT_EXTS), help="Estensioni valide (lowercase)")
    p.add_argument("--workers", type=int, default=1, help="Worker paralleli per il resize")
    p.add_argument("--dry-run", action="store_true", help="Non scrive file, ma produce il piano e i CSV")
    p.add_argument(
        "--meta-csv",
        type=Path,
        default=None,
        help="Percorso per letterbox_meta.csv (default: OUT_DIR/letterbox_meta.csv)",
    )
    p.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Percorso per manifest_letterboxed.csv (default: OUT_DIR/manifest_letterboxed.csv)",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    time_re = re.compile(args.time_regex)
    fname_formats = list(args.strptime)
    csv_formats = list(args.csv_strptime)
    exts = set(e.lower() for e in args.ext)
    buffer = timedelta(hours=args.buffer_hours)

    meta_csv = args.meta_csv or (args.out_dir / "letterbox_meta.csv")
    manifest_csv = args.manifest_csv or (args.out_dir / "manifest_letterboxed.csv")
    images_root = args.out_dir  # salviamo direttamente qui, senza aggiungere un ulteriore "resized"

    print(
        "[ARGS]",
        f"windows_csv={args.windows_csv}",
        f"src={args.src}",
        f"out_dir={args.out_dir}",
        f"image_size={args.image_size}",
        f"resize_mode={args.resize_mode}",
        f"buffer_hours={args.buffer_hours}",
        f"recurse={args.recurse}",
        f"preserve_structure={args.preserve_structure}",
        f"workers={args.workers}",
        f"dry_run={args.dry_run}",
        f"time_regex={args.time_regex}",
        f"strptime={fname_formats}",
        f"csv_strptime={csv_formats}",
        f"exts={sorted(exts)}",
        f"meta_csv={meta_csv}",
        f"manifest_csv={manifest_csv}",
        f"images_root={images_root}",
        flush=True,
    )

    # Validate inputs
    if not args.windows_csv.exists():
        raise FileNotFoundError(args.windows_csv)
    if not args.src.exists():
        raise FileNotFoundError(args.src)

    # Load windows
    windows = load_windows(args.windows_csv, buffer, buffer, csv_formats, args.id_cols)
    if not windows:
        print("No windows found in CSV.")
        return 1

    ensure_dir(args.out_dir)
    ensure_dir(images_root)

    # Scan files
    files = iter_files(args.src, args.recurse, exts)

    # Prepare plans
    plans: List[Tuple[Path, Path, str, int, str, int]] = []

    scanned = 0
    for p in files:
        scanned += 1
        if scanned % 5000 == 0:
            print(f"[SCAN] scanned {scanned} files so far...", flush=True)
        dt = parse_dt_from_filename(p.name, time_re, fname_formats)
        if dt is None:
            continue
        in_ext_any = False
        in_core_any = False
        ids_ext: List[str] = []
        ids_core: List[str] = []
        for w in windows:
            if w.start_ext <= dt <= w.end_ext:
                in_ext_any = True
                ids_ext.append(w.event_id)
                if w.start_core <= dt <= w.end_core:
                    in_core_any = True
                    ids_core.append(w.event_id)
        if not in_ext_any:
            continue
        label = 1 if in_core_any else 0
        event_ids = "|".join(sorted(set(ids_core if label == 1 else ids_ext)))
        rel = p.relative_to(args.src) if args.preserve_structure else Path(p.name)
        dst_path = images_root / rel
        plans.append(
            (
                p,
                dst_path,
                dt.isoformat(timespec="minutes"),
                label,
                event_ids,
                1 if in_core_any else 0,
            )
        )

    print(f"[INFO] scanned {scanned} files; planned {len(plans)} to process.", flush=True)
    if not plans:
        print("[INFO] Nessun file nel buffer/finestra (controlla regex, estensioni, buffer-hours).", flush=True)

    meta_rows: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []

    def process_plan(plan: Tuple[Path, Path, str, int, str, int]) -> Tuple[Dict[str, object], Dict[str, object]]:
        src_path, dst_path, dt_iso, label, event_ids, in_core = plan
        meta_row: Dict[str, object] = {
            "orig_path": str(src_path),
            "resized_path": str(dst_path),
            "datetime_iso": dt_iso,
            "label": label,
            "in_core_window": in_core,
            "in_extended_window": 1,
            "event_ids": event_ids,
            "resize_mode": args.resize_mode,
            "orig_w": "",
            "orig_h": "",
            "out_size": args.image_size,
            "w_new": "",
            "h_new": "",
            "scale": "",
            "scale_x": "",
            "scale_y": "",
            "pad_x": "",
            "pad_y": "",
            "status": "planned",
            "status_msg": "",
        }
        manifest_row: Dict[str, object] = {
            "src_path": str(src_path),
            "dst_path": str(dst_path),
            "datetime_iso": dt_iso,
            "label": label,
            "in_core_window": in_core,
            "in_extended_window": 1,
            "event_ids": event_ids,
            "status": "planned",
            "status_msg": "",
        }
        try:
            img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"cannot read image: {src_path}")
            H, W = img.shape[:2]
            resized, meta = resize_image(img, args.image_size, mode=args.resize_mode)
            meta_row.update(
                {
                    "orig_w": W,
                    "orig_h": H,
                    "w_new": meta["w_new"],
                    "h_new": meta["h_new"],
                    "scale": meta["scale"],
                    "scale_x": meta.get("scale_x", meta["scale"]),
                    "scale_y": meta.get("scale_y", meta["scale"]),
                    "pad_x": meta["pad_x"],
                    "pad_y": meta["pad_y"],
                }
            )
            if args.dry_run:
                status = "dry-run"
                msg = "not written"
                print(f"[PLAN] {dst_path}", flush=True)
            else:
                ensure_dir(dst_path.parent)
                if dst_path.exists():
                    status = "exists"
                    msg = "already exists"
                else:
                    if not cv2.imwrite(str(dst_path), resized):
                        raise IOError(f"cannot write: {dst_path}")
                    status = "written"
                    msg = "ok"
        except Exception as e:
            status = "error"
            msg = str(e)
        meta_row["status"] = status
        meta_row["status_msg"] = msg
        manifest_row["status"] = status
        manifest_row["status_msg"] = msg
        return meta_row, manifest_row

    completed = 0

    def log_progress():
        if completed and completed % 5000 == 0:
            print(f"[WRITE] completed {completed} / {len(plans)}", flush=True)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_plan, plan): plan for plan in plans}
            for fut in as_completed(futs):
                meta_row, manifest_row = fut.result()
                meta_rows.append(meta_row)
                manifest_rows.append(manifest_row)
                completed += 1
                log_progress()
    else:
        for plan in plans:
            meta_row, manifest_row = process_plan(plan)
            meta_rows.append(meta_row)
            manifest_rows.append(manifest_row)
            completed += 1
            log_progress()

    # Write CSVs
    if meta_rows:
        ensure_dir(meta_csv.parent)
        meta_fields = [
            "orig_path",
            "resized_path",
            "datetime_iso",
            "label",
            "in_core_window",
            "in_extended_window",
            "event_ids",
            "resize_mode",
            "orig_w",
            "orig_h",
            "out_size",
            "w_new",
            "h_new",
            "scale",
            "scale_x",
            "scale_y",
            "pad_x",
            "pad_y",
            "status",
            "status_msg",
        ]
        with meta_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=meta_fields)
            w.writeheader()
            w.writerows(meta_rows)

    if manifest_rows:
        ensure_dir(manifest_csv.parent)
        manifest_fields = [
            "src_path",
            "dst_path",
            "datetime_iso",
            "label",
            "in_core_window",
            "in_extended_window",
            "event_ids",
            "status",
            "status_msg",
        ]
        with manifest_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=manifest_fields)
            w.writeheader()
            w.writerows(manifest_rows)

    # Summary
    total = len(manifest_rows)
    written = sum(1 for r in manifest_rows if r.get("status") == "written")
    dry = sum(1 for r in manifest_rows if r.get("status") == "dry-run")
    errors = sum(1 for r in manifest_rows if r.get("status") == "error")
    pos = sum(1 for r in manifest_rows if r.get("label") == 1)
    neg = total - pos
    print(f"Planned {len(plans)}; wrote {total} manifest rows.")
    print(f"Pos {pos} / Neg {neg}; written {written}, dry-run {dry}, errors {errors}.")
    if args.dry_run:
        print("[dry-run] No images were written.")
    else:
        print(f"Letterboxed images under: {images_root}")
        print(f"letterbox_meta: {meta_csv}")
        print(f"manifest:      {manifest_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
