#!/usr/bin/env python3
r"""Copy a labelling set of standing frames out of the raw folders.

Picks standing frames from posture predictions (select_frames.py rules: posture
confidence, time windows, at most N per sow per day spread over the day) and
copies each IR image and its depth file into OUT/<timestamp_folder>/. Raw folders
are only read. OUT/label_set.csv lists the frames with paths relative to OUT,
so the set can be labelled on any computer (label_ellipse.py) and trained on
anywhere (train_seg.py resolves relative paths against the labels CSV folder).

Usage (lab PC, copying to the Mac through the Remote Desktop share):
    python vulva/export_label_set.py outputs\gt_feb06\predictions.csv \\tsclient\paal_exchange\vulva_label_set ^
        --windows 06:00-11:00,14:00-18:00 --per-day 3
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_frames import parse_windows, select  # noqa: E402

FIELDS = ["timestamp_folder", "pig_id", "pig_timestamp", "ir_path", "depth_path", "posture_confidence"]


def export(pred_rows, out_dir, windows=None, per_day=3, min_conf=0.6, dates=None):
    picked = select(pred_rows, min_conf, windows, per_day)
    if dates:
        picked = [p for p in picked if p["pig_timestamp"][:8] in dates]
    os.makedirs(out_dir, exist_ok=True)
    rows, missing = [], 0
    for p in picked:
        src_ir, src_d = p["ir_path"], p["depth_path"]
        if not os.path.exists(src_ir):
            missing += 1
            continue
        sub = os.path.join(out_dir, p["timestamp_folder"])
        os.makedirs(sub, exist_ok=True)
        ir_rel = os.path.join(p["timestamp_folder"], os.path.basename(src_ir))
        shutil.copy2(src_ir, os.path.join(out_dir, ir_rel))
        d_rel = ""
        if os.path.exists(src_d):
            d_rel = os.path.join(p["timestamp_folder"], os.path.basename(src_d))
            shutil.copy2(src_d, os.path.join(out_dir, d_rel))
        rows.append({**p, "ir_path": ir_rel.replace("\\", "/"), "depth_path": d_rel.replace("\\", "/")})
    with open(os.path.join(out_dir, "label_set.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", help="posture predictions.csv")
    ap.add_argument("out_dir")
    ap.add_argument("--windows", default="06:00-11:00,14:00-18:00")
    ap.add_argument("--per-day", type=int, default=3, help="frames per sow per day")
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--dates", default="", help="comma list of YYYYMMDD to keep (default: all)")
    args = ap.parse_args()

    with open(args.predictions, newline="", encoding="utf-8-sig") as f:
        preds = list(csv.DictReader(f))
    dates = {d.strip() for d in args.dates.split(",") if d.strip()} or None
    rows, missing = export(preds, args.out_dir, parse_windows(args.windows), args.per_day, args.min_conf, dates)
    sows = {r["pig_id"] for r in rows}
    days = {r["pig_timestamp"][:8] for r in rows}
    print(f"Copied {len(rows)} frames ({len(sows)} sows, {len(days)} days) to {args.out_dir}; "
          f"{missing} IR files not found")


if __name__ == "__main__":
    main()
