#!/usr/bin/env python3
"""Pick the frames worth measuring: the sow is standing, by the posture model.

The vulva is only visible from behind when the sow stands (lying hides it,
sitting squashes it). This reads predictions.csv from posture/run_inference.py
and keeps standing frames, optionally only inside time windows (feeding times,
when most sows stand) and at most N frames per sow per day.

Usage:
    python vulva/select_frames.py outputs/predictions.csv --windows 06:00-11:00,14:00-18:00
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

FIELDS = ["timestamp_folder", "pig_id", "pig_timestamp", "ir_path", "depth_path", "posture_confidence"]
BAD_ID = ("overflow", "duplicate", "dropped")


def parse_windows(spec):
    """'06:00-11:00,14:00-18:00' -> [(360, 660), (840, 1080)] in minutes."""
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("-")
        to_min = lambda s: int(s.split(":")[0]) * 60 + int(s.split(":")[1])  # noqa: E731
        out.append((to_min(a), to_min(b)))
    return out


def minute_of_day(ts):
    """'20260211-09-07-10' -> 547."""
    try:
        return int(ts[9:11]) * 60 + int(ts[12:14])
    except (ValueError, IndexError):
        return None


def depth_path_for(ir_path):
    d, name = os.path.split(ir_path)
    return os.path.join(d, name.replace("_ir_vis_", "_depth_").rsplit(".", 1)[0] + ".raw")


def select(pred_rows, min_conf=0.6, windows=None, max_per_day=0):
    picked = []
    for r in pred_rows:
        if (r.get("prediction_name") or "").strip() != "standing":
            continue
        try:
            conf = float(r.get("confidence") or 0)
        except ValueError:
            conf = 0.0
        if conf < min_conf:
            continue
        if any(b in (r.get("id_status") or "") for b in BAD_ID):
            continue
        ts = r.get("pig_timestamp") or r.get("timestamp_folder") or ""
        if windows:
            m = minute_of_day(ts)
            if m is None or not any(a <= m <= b for a, b in windows):
                continue
        ir = r.get("image_path") or ""
        if not ir:
            continue
        picked.append({"timestamp_folder": r["timestamp_folder"], "pig_id": int(float(r["pig_id"])),
                       "pig_timestamp": ts, "ir_path": ir, "depth_path": depth_path_for(ir),
                       "posture_confidence": round(conf, 4)})
    if max_per_day:
        by = defaultdict(list)
        for p in picked:
            by[(p["pig_id"], p["pig_timestamp"][:8])].append(p)
        thinned = []
        for rs in by.values():
            rs.sort(key=lambda p: p["pig_timestamp"])
            if len(rs) > max_per_day:        # spread over the day, centred (1 per day = the middle one)
                step = len(rs) / max_per_day
                rs = [rs[int((i + 0.5) * step)] for i in range(max_per_day)]
            thinned.extend(rs)
        picked = thinned
    return sorted(picked, key=lambda p: (p["timestamp_folder"], p["pig_id"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", help="predictions.csv from posture/run_inference.py")
    ap.add_argument("--min-conf", type=float, default=0.6)
    ap.add_argument("--windows", default="", help="e.g. 06:00-11:00,14:00-18:00 (default: all day)")
    ap.add_argument("--max-per-day", type=int, default=0, help="0 = no limit")
    ap.add_argument("--out", default=os.path.join("outputs", "vulva", "candidates.csv"))
    args = ap.parse_args()

    with open(args.predictions, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    picked = select(rows, args.min_conf, parse_windows(args.windows), args.max_per_day)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(picked)
    days = {(p["pig_id"], p["pig_timestamp"][:8]) for p in picked}
    print(f"{len(picked)} standing frames from {len(rows)} predictions "
          f"({len(days)} sow-days, {len(picked) / max(len(days), 1):.1f} per sow-day) -> {args.out}")


if __name__ == "__main__":
    main()
