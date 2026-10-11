#!/usr/bin/env python3
"""How noisy are per-frame vulva measurements? Run this before trusting any estrus signal.

Within one morning the vulva does not change, so the spread of one sow's
measurements across passes on the same day is pure measurement noise. This
script reports, per sow and overall:
  - within-day spread (SD) of width and length,
  - error against Lucas's caliper values for that day (if --truth is given),
  - how many frames per sow-day a daily median needs to get its standard
    error under --target mm (n = (SD / target)^2).

Works on any per-frame CSV with timestamp_folder, pig_id, a status column
(ok / measured) and width_mm, length_mm: the output of run_vulva.py or the
manual vulva_measurements_manual_dataset.csv.

Usage:
    python vulva/repeatability.py vulva_measurements_manual_dataset.csv --truth outputs/vulva/sow_day_truth.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from daily import QUALITY, _f, frame_date, is_good  # noqa: E402


def load_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def analyse(rows, truth=None, target_mm=2.0, quality=QUALITY):
    all_frames = defaultdict(int)
    good = defaultdict(list)
    for r in rows:
        key = (int(float(r["pig_id"])), frame_date(r))
        all_frames[key] += 1
        if is_good(r, quality):
            good[key].append(r)
    t = {(int(x["pig_id"]), x["date"]): x for x in (truth or [])}
    per, sds = [], {"width_mm": [], "length_mm": []}
    errs = {"width_mm": [], "length_mm": []}
    for key in sorted(all_frames):
        rs = good.get(key, [])
        rec = {"pig_id": key[0], "date": key[1], "frames": all_frames[key], "usable": len(rs)}
        cal = t.get(key)
        for k, ck in (("width_mm", "vulva_width_mm"), ("length_mm", "vulva_length_mm")):
            vals = [_f(r[k]) for r in rs]
            if vals:
                rec[k + "_median"] = round(statistics.median(vals), 1)
            if len(vals) >= 2:
                sd = statistics.stdev(vals)
                rec[k + "_sd"] = round(sd, 2)
                if len(vals) >= 3:
                    sds[k].append(sd)
            if cal and vals and _f(cal.get(ck)) is not None:
                rec[k + "_caliper"] = _f(cal[ck])
                errs[k].extend(v - _f(cal[ck]) for v in vals)
        per.append(rec)
    summary = {"sow_days": len(per), "frames": sum(all_frames.values()),
               "usable_frames": sum(len(v) for v in good.values())}
    for k in ("width_mm", "length_mm"):
        if sds[k]:
            med = statistics.median(sds[k])
            summary[k + "_within_day_sd"] = round(med, 2)
            summary[k + "_frames_needed"] = math.ceil((med / target_mm) ** 2)
        if errs[k]:
            summary[k + "_mae_vs_caliper"] = round(statistics.mean(abs(e) for e in errs[k]), 2)
            summary[k + "_bias_vs_caliper"] = round(statistics.mean(errs[k]), 2)
    return per, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("measurements", help="per-frame CSV")
    ap.add_argument("--truth", help="sow_day_truth.csv from vulva/truth.py")
    ap.add_argument("--target", type=float, default=2.0, help="target standard error of a daily median, mm")
    ap.add_argument("--out", default=os.path.join("outputs", "vulva", "repeatability.csv"))
    args = ap.parse_args()

    rows = load_csv(args.measurements)
    truth = load_csv(args.truth) if args.truth else None
    per, s = analyse(rows, truth, args.target)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fields = sorted({k for r in per for k in r}, key=lambda k: (k not in ("pig_id", "date", "frames", "usable"), k))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per)

    print(f"{s['frames']} frames, {s['usable_frames']} usable, {s['sow_days']} sow-days")
    for k, name in (("width_mm", "Width"), ("length_mm", "Length")):
        if k + "_within_day_sd" in s:
            print(f"{name}: within-day SD {s[k + '_within_day_sd']} mm (median over sow-days with 3+ frames); "
                  f"{s[k + '_frames_needed']} frames per sow-day for a {args.target} mm standard error")
        if k + "_mae_vs_caliper" in s:
            print(f"{name}: error vs caliper {s[k + '_mae_vs_caliper']} mm MAE, bias {s[k + '_bias_vs_caliper']} mm")
    print(f"Per sow-day table -> {args.out}")


if __name__ == "__main__":
    main()
