#!/usr/bin/env python3
"""Batch-build manual vulva depth outputs from saved IR/depth-space polygons.

Reads labels/vulva_manual_labels.csv, loads each polygon mask, and saves
mask/surface depth outputs under each frame's depthmap folder.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys


print(
    "The old vulva measurement workflow was removed. "
    "Use `python label_vulva_dataset.py` and click 2 points for length plus 2 points for width on the IR image."
)
raise SystemExit(0)

import cv2
import numpy as np

from config import VULVA_MANUAL_LABELS_CSV
from measure_vulva_manual import load_frame_context, process_vulva_measurement, repo_root


DEFAULT_OUTPUT_CSV = repo_root() / "labels" / "vulva_depthmap_manual_dataset.csv"
CSV_FIELDS = [
    "timestamp_folder",
    "pig_id",
    "pig_timestamp",
    "status",
    "depthmap_dir",
    "depth_raw",
    "texture_input",
    "mask_path",
    "focus_method",
    "focus_box",
    "stretched_mask_path",
    "no_vulva_surface_path",
    "vulva_only_surface_path",
    "panel_path",
    "summary_path",
]


def resolve_repo_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root() / path


def load_label_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Label CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def measurement_focus_box(mask: np.ndarray, margin: int = 24):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("Manual mask is empty")
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(mask.shape[1] - 1, int(xs.max()) + margin)
    y2 = min(mask.shape[0] - 1, int(ys.max()) + margin)
    return (x1, y1, x2, y2)


def write_output_rows(output_csv: Path, rows: list[dict]):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def main(args):
    label_rows = load_label_rows(Path(args.labels_csv))
    if args.pig is not None:
        label_rows = [row for row in label_rows if int(row["pig_id"]) == args.pig]

    polygon_total = sum(1 for row in label_rows if row.get("status") == "polygon")
    print(f"Rows in label CSV: {len(label_rows)}")
    print(f"Polygon rows queued for depth outputs: {polygon_total}")
    print("Batch mode: this script builds depth outputs only and does not open an annotation window.")

    results = []
    processed = 0
    status_counts = Counter()
    polygon_index = 0
    for row in label_rows:
        base = {
            "timestamp_folder": row.get("timestamp_folder", ""),
            "pig_id": row.get("pig_id", ""),
            "pig_timestamp": row.get("pig_timestamp", ""),
            "status": row.get("status", ""),
            "depthmap_dir": row.get("depthmap_dir", ""),
            "depth_raw": row.get("depth_raw", ""),
            "mask_path": row.get("mask_path", ""),
        }

        if row.get("status") != "polygon":
            results.append(base)
            status_counts[base["status"] or "blank_status"] += 1
            continue

        polygon_index += 1
        frame_tag = f"[{polygon_index}/{polygon_total}] {base['timestamp_folder']} pig{base['pig_id']}"
        print(f"{frame_tag} building depth outputs...", flush=True)

        depthmap_dir = resolve_repo_path(row.get("depthmap_dir", ""))
        depth_raw = resolve_repo_path(row.get("depth_raw", ""))
        frame_input = depthmap_dir if depthmap_dir and depthmap_dir.exists() else depth_raw
        if frame_input is None or not frame_input.exists():
            base["status"] = "missing_frame_input"
            results.append(base)
            status_counts[base["status"]] += 1
            print(f"{frame_tag} -> missing_frame_input", flush=True)
            continue

        mask_path = resolve_repo_path(row.get("mask_path", ""))
        if mask_path is None or not mask_path.exists():
            base["status"] = "missing_mask"
            results.append(base)
            status_counts[base["status"]] += 1
            print(f"{frame_tag} -> missing_mask", flush=True)
            continue

        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            base["status"] = "unreadable_mask"
            results.append(base)
            status_counts[base["status"]] += 1
            print(f"{frame_tag} -> unreadable_mask", flush=True)
            continue

        ctx = load_frame_context(frame_input)
        mask_bool = raw_mask > 0
        if mask_bool.shape != ctx["depth_mm"].shape:
            base["status"] = "shape_mismatch"
            results.append(base)
            status_counts[base["status"]] += 1
            print(f"{frame_tag} -> shape_mismatch", flush=True)
            continue

        focus_box = measurement_focus_box(mask_bool)
        output_dir = (repo_root() / row["depthmap_dir"]) / "vulva_manual" if row.get("depthmap_dir") else Path(ctx["frame_dir"]) / "vulva_manual"

        try:
            summary = process_vulva_measurement(
                ctx=ctx,
                raw_mask=mask_bool,
                focus_box=focus_box,
                focus_method="manual_polygon",
                output_dir=output_dir,
                csv_path=None,
            )
        except Exception as exc:
            base["status"] = f"failed: {exc}"
            results.append(base)
            status_counts["failed"] += 1
            print(f"{frame_tag} -> failed: {exc}", flush=True)
            continue

        processed += 1
        results.append(
            {
                **base,
                "status": "built_depth_outputs",
                "depth_raw": summary.get("depth_raw", base["depth_raw"]),
                "texture_input": summary.get("texture_input", ""),
                "focus_method": summary.get("focus_method", ""),
                "focus_box": json.dumps(summary.get("focus_box", [])),
                "stretched_mask_path": summary.get("stretched_mask_path", ""),
                "no_vulva_surface_path": summary.get("no_vulva_surface_path", ""),
                "vulva_only_surface_path": summary.get("vulva_only_surface_path", ""),
                "panel_path": summary.get("panel_path", ""),
                "summary_path": str(output_dir / "vulva_depthmap_summary.json"),
            }
        )
        status_counts["built_depth_outputs"] += 1
        print(f"{frame_tag} -> built depth outputs", flush=True)

    output_csv = Path(args.output_csv)
    write_output_rows(output_csv, results)
    print(f"Polygon rows converted to depth outputs: {processed}")
    print(f"Status counts: {dict(status_counts)}")
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-build manual vulva depth outputs from saved polygon masks")
    parser.add_argument("--labels-csv", default=VULVA_MANUAL_LABELS_CSV, help="Manual label CSV from label_vulva_dataset.py")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Where to write the dataset-wide depth-output manifest")
    parser.add_argument("--pig", type=int, default=None, help="Only build depth outputs for this pig ID")
    main(parser.parse_args())
