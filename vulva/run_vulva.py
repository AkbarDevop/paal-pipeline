#!/usr/bin/env python3
"""One-command vulva pipeline: standing frames -> masks -> mm -> one value per sow per day.

  1. Pick standing frames from the posture predictions (select_frames.py).
  2. Segment the vulva on each IR frame (models/vulva_seg.pth, train_seg.py).
  3. Measure width, length, height and volume from the matching depth frame
     (measure3d.py), with quality numbers per frame.
  4. Daily median per sow and % change from her own first days (daily.py).
  5. With --truth, compare with Lucas's caliper values and heat days.

Raw folders are only read. Everything is written under --out-dir.

Usage:
    python vulva/run_vulva.py outputs/gt_feb06/predictions.csv --truth outputs/vulva/sow_day_truth.csv
    python vulva/run_vulva.py predictions.csv --windows 06:00-11:00,14:00-18:00 --overlays 60
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from daily import METRICS, add_baseline_change, aggregate, compare_truth  # noqa: E402
from measure3d import MEASURE_FIELDS, load_depth_raw, load_intrinsics, measure  # noqa: E402
from seg_model import ir_channel, load_checkpoint, predict, read_input  # noqa: E402
from select_frames import parse_windows, select  # noqa: E402

FRAME_FIELDS = ["timestamp_folder", "pig_id", "pig_timestamp", "posture_confidence",
                "vulva_visible", "vulva_prob", "vulva_conf", "mask_area_px"] + MEASURE_FIELDS + ["ir_path", "mask_path"]


def write_csv(rows, path, fields=None):
    fields = fields or sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def overlay_tile(gray, mask, label, size=160):
    ys, xs = np.nonzero(mask)
    cy, cx = int(ys.mean()), int(xs.mean())
    h = 50
    y1, y2 = max(cy - h, 0), min(cy + h, gray.shape[0])
    x1, x2 = max(cx - h, 0), min(cx + h, gray.shape[1])
    tile = cv2.cvtColor(gray[y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)
    cnts, _ = cv2.findContours(mask[y1:y2, x1:x2].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(tile, cnts, -1, (0, 220, 255), 1)
    tile = cv2.resize(tile, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.putText(tile, label, (4, size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def save_sheet(tiles, path, cols=8):
    if not tiles:
        return
    blank = np.zeros_like(tiles[0])
    rows = [tiles[i:i + cols] + [blank] * (cols - len(tiles[i:i + cols])) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack([np.hstack(r) for r in rows]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", help="predictions.csv from posture/run_inference.py")
    ap.add_argument("--model", default=os.path.join("models", "vulva_seg.pth"))
    ap.add_argument("--out-dir", default=os.path.join("outputs", "vulva"))
    ap.add_argument("--min-conf", type=float, default=0.6, help="minimum posture confidence for 'standing'")
    ap.add_argument("--windows", default="", help="time windows, e.g. 06:00-11:00,14:00-18:00")
    ap.add_argument("--max-per-day", type=int, default=0, help="frames per sow-day, 0 = all")
    ap.add_argument("--truth", default="", help="sow_day_truth.csv from vulva/truth.py")
    ap.add_argument("--calibration", default="", help="calibration.json with tof_intrinsics")
    ap.add_argument("--baseline-days", type=int, default=2)
    ap.add_argument("--min-frames", type=int, default=3, help="minimum good frames for a daily value")
    ap.add_argument("--overlays", type=int, default=48, help="frames on the review contact sheet")
    ap.add_argument("--save-masks", action="store_true", help="write each predicted mask as PNG")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.model, device)
    thr, min_area = ckpt.get("threshold", 0.5), ckpt.get("min_area", 25)
    presence = ckpt.get("presence_threshold", 0.0)   # models before r5 have none
    inputs = ckpt.get("inputs", "ir")
    K = load_intrinsics(args.calibration)

    with open(args.predictions, newline="", encoding="utf-8-sig") as f:
        preds = list(csv.DictReader(f))
    cands = select(preds, args.min_conf, parse_windows(args.windows), args.max_per_day)
    print(f"{len(cands)} standing frames selected from {len(preds)} predictions")

    mask_dir = os.path.join(args.out_dir, "masks")
    if args.save_masks:
        os.makedirs(mask_dir, exist_ok=True)
    rows, tiles = [], []
    for i in range(0, len(cands), 64):
        chunk = cands[i:i + 64]
        grays = [read_input(c["ir_path"], c["depth_path"], inputs) for c in chunk]
        ok = [j for j, g in enumerate(grays) if g is not None]
        res = predict(model, [grays[j] for j in ok], device, thr, min_area, args.batch_size, presence)
        res_by = dict(zip(ok, res))
        for j, c in enumerate(chunk):
            row = {k: c[k] for k in ("timestamp_folder", "pig_id", "pig_timestamp", "posture_confidence", "ir_path")}
            if j not in res_by:
                row.update({"vulva_visible": "", "measure_status": "unreadable_ir"})
                rows.append(row)
                continue
            mask, prob, area, conf = res_by[j]
            row.update({"vulva_visible": int(mask.any()), "vulva_prob": round(prob, 4),
                        "vulva_conf": round(conf, 4), "mask_area_px": area})
            if mask.any():
                m = measure(load_depth_raw(c["depth_path"]), mask, K)
                row.update(m)
                if args.save_masks:
                    p = os.path.join(mask_dir, f"{c['timestamp_folder']}_pig{c['pig_id']}_{c['pig_timestamp']}.png")
                    cv2.imwrite(p, mask * 255)
                    row["mask_path"] = p
                if len(tiles) < args.overlays and m["measure_status"] == "ok":
                    tiles.append(overlay_tile(ir_channel(grays[j]), mask, f"p{c['pig_id']} {c['pig_timestamp'][4:13]} "
                                              f"W{m['width_mm']:.0f} L{m['length_mm']:.0f}"))
            else:
                row["measure_status"] = "low_confidence" if 0 < conf < presence else "not_visible"
            rows.append(row)
        print(f"  {min(i + 64, len(cands))}/{len(cands)} frames", flush=True)

    frames_csv = os.path.join(args.out_dir, "vulva_frames.csv")
    write_csv(rows, frames_csv, FRAME_FIELDS)
    save_sheet(tiles, os.path.join(args.out_dir, "vulva_review.jpg"))

    daily = add_baseline_change(aggregate(rows, min_frames=args.min_frames), args.baseline_days)
    daily_fields = ["pig_id", "date", "n_frames"] + [k + s for k in METRICS for s in ("", "_mad", "_change_pct")]
    write_csv(daily, os.path.join(args.out_dir, "vulva_daily.csv"), daily_fields)

    visible = sum(1 for r in rows if r.get("vulva_visible") == 1)
    good = sum(1 for r in rows if r.get("measure_status") == "ok")
    unsure = sum(1 for r in rows if r.get("measure_status") == "low_confidence")
    summary = {"frames": len(rows), "visible": visible, "low_confidence": unsure, "measured_ok": good,
               "sow_days": len(daily), "model": args.model, "presence_threshold": presence,
               "seconds": round(time.time() - t0, 1)}
    print(f"Vulva visible in {visible} of {len(rows)} standing frames ({unsure} more dropped as low confidence); "
          f"measured {good}; {len(daily)} sow-days with at least {args.min_frames} good frames")

    if args.truth:
        with open(args.truth, newline="", encoding="utf-8-sig") as f:
            truth = list(csv.DictReader(f))
        joined, s = compare_truth(daily, truth)
        write_csv(joined, os.path.join(args.out_dir, "vulva_vs_caliper.csv"))
        summary["vs_caliper"] = s
        print(f"Against calipers ({s['sow_days']} sow-days):")
        for k, name in (("width_mm", "width"), ("length_mm", "length"), ("vulva_index", "W x L / 200")):
            if k + "_mae" in s:
                print(f"  {name}: MAE {s[k + '_mae']}, bias {s[k + '_bias']}, r {s[k + '_r']}")
        if "index_change_heat_days" in s:
            print(f"  W x L change from baseline: heat days {s['index_change_heat_days']}%, "
                  f"other days {s['index_change_other_days']}%")
    with open(os.path.join(args.out_dir, "vulva_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Outputs in {args.out_dir} ({summary['seconds']} s)")


if __name__ == "__main__":
    main()
