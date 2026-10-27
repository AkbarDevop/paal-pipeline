#!/usr/bin/env python3
r"""Copy a labelling set of standing frames out of the raw folders.

Picks standing frames from posture predictions (select_frames.py rules: posture
confidence, time windows, at most N per sow per day spread over the day) and
copies each IR image and its depth file into OUT/<timestamp_folder>/. Raw folders
are only read. OUT/label_set.csv lists the frames with paths relative to OUT,
so the set can be labelled on any computer (label_ellipse.py) and trained on
anywhere (train_seg.py resolves relative paths against the labels CSV folder).

With --rgb the colour images are copied too: pigN_rgb_aligned_<ts> (aligned
to the ToF frame, so it can be clicked like the IR image) and pigN_rgb_<ts>
(separate camera, not aligned; shown for reference). --add-rgb adds them to an
existing set without changing its frames or labels.

Usage (lab PC, copying to the Mac through the Remote Desktop share):
    python vulva/export_label_set.py outputs\gt_feb06\predictions.csv \\tsclient\paal_exchange\vulva_label_set ^
        --windows 06:00-11:00,14:00-18:00 --per-day 3
    python vulva/export_label_set.py outputs\gt_feb06\predictions.csv \\tsclient\paal_exchange\vulva_label_set --add-rgb
    # test set for eval_seg.py: new days, 2 per sow-day spread over the whole day, labelled without --model
    python vulva/export_label_set.py outputs\gt_feb12\predictions.csv \\tsclient\paal_exchange\vulva_test_feb12 ^
        --dates 20260212,20260213,20260214,20260215 --per-day 2 --windows 00:00-23:59 --rgb
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
RGB_FIELDS = ["rgb_aligned_path", "rgb_path"]
RGB_KINDS = (("rgb_aligned_path", "_rgb_aligned_"), ("rgb_path", "_rgb_"))


def rgb_sources(ir_path):
    """Colour images next to an IR frame: {field: existing path} for pigN_rgb_aligned_<ts>
    and pigN_rgb_<ts> (.jpg or .png)."""
    d, name = os.path.split(ir_path)
    stem = name.rsplit(".", 1)[0]
    found = {}
    for field, tag in RGB_KINDS:
        for ext in (".jpg", ".png", ".jpeg"):
            cand = os.path.join(d, stem.replace("_ir_vis_", tag) + ext)
            if os.path.exists(cand):
                found[field] = cand
                break
    return found


def _copy_rgb(ir_src, out_dir, folder):
    rel = {}
    for field, src in rgb_sources(ir_src).items():
        r = os.path.join(folder, os.path.basename(src))
        shutil.copy2(src, os.path.join(out_dir, r))
        rel[field] = r.replace("\\", "/")
    return rel


def export(pred_rows, out_dir, windows=None, per_day=3, min_conf=0.6, dates=None, rgb=False):
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
        row = {**p, "ir_path": ir_rel.replace("\\", "/"), "depth_path": d_rel.replace("\\", "/")}
        if rgb:
            row.update(_copy_rgb(src_ir, out_dir, p["timestamp_folder"]))
        rows.append(row)
    with open(os.path.join(out_dir, "label_set.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + (RGB_FIELDS if rgb else []), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows, missing


def add_rgb(pred_rows, out_dir):
    """Copy the colour images for the frames already in OUT/label_set.csv and add their
    columns. The frame list, IR, depth and any labels.csv are left as they are."""
    set_csv = os.path.join(out_dir, "label_set.csv")
    with open(set_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)
    src_by = {}
    for r in pred_rows:
        ir = r.get("image_path") or ""
        if ir:
            src_by[(r["timestamp_folder"], os.path.basename(ir))] = ir
    counts = {k: 0 for k in RGB_FIELDS}
    no_source = 0
    for row in rows:
        src = src_by.get((row["timestamp_folder"], os.path.basename(row["ir_path"].replace("/", os.sep))))
        if not src:
            no_source += 1
            continue
        rel = _copy_rgb(src, out_dir, row["timestamp_folder"])
        for k in RGB_FIELDS:
            row[k] = rel.get(k, row.get(k, ""))
            counts[k] += bool(row[k])
    for k in RGB_FIELDS:
        if k not in fields:
            fields.append(k)
    tmp = set_csv + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, set_csv)
    return len(rows), counts, no_source


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", help="posture predictions.csv")
    ap.add_argument("out_dir")
    ap.add_argument("--windows", default="06:00-11:00,14:00-18:00")
    ap.add_argument("--per-day", type=int, default=3, help="frames per sow per day")
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--dates", default="", help="comma list of YYYYMMDD to keep (default: all)")
    ap.add_argument("--rgb", action="store_true", help="also copy the colour images (rgb_aligned and rgb)")
    ap.add_argument("--add-rgb", action="store_true",
                    help="only add colour images to the existing set in OUT (frames and labels unchanged)")
    args = ap.parse_args()

    with open(args.predictions, newline="", encoding="utf-8-sig") as f:
        preds = list(csv.DictReader(f))
    if args.add_rgb:
        n, counts, no_source = add_rgb(preds, args.out_dir)
        print(f"{n} frames in {args.out_dir}: aligned RGB for {counts['rgb_aligned_path']}, "
              f"plain RGB for {counts['rgb_path']}; {no_source} frames not found in the predictions")
        return
    dates = {d.strip() for d in args.dates.split(",") if d.strip()} or None
    rows, missing = export(preds, args.out_dir, parse_windows(args.windows), args.per_day, args.min_conf, dates,
                           args.rgb)
    sows = {r["pig_id"] for r in rows}
    days = {r["pig_timestamp"][:8] for r in rows}
    print(f"Copied {len(rows)} frames ({len(sows)} sows, {len(days)} days) to {args.out_dir}; "
          f"{missing} IR files not found")


if __name__ == "__main__":
    main()
