#!/usr/bin/env python3
"""Build the sow-day table the heat model learns from.

One row per sow per day D, joining:
  - posture features for the night that ends on the morning of D
    (19:00 on D-1 to 08:00 on D, the same convention as count_posture_changes.py)
    and for the daytime of D (08:00-19:00), from posture/run_inference.py predictions;
  - days since the sows were moved in (Lucas's DAY column, 0 = MOVED), as the
    days-from-weaning input;
  - heat on D (morning or evening check, Lucas's sheet via vulva/truth.py);
  - optionally the daily vulva values from vulva/run_vulva.py.
Each posture and vulva feature also gets a "_delta" column: the change from
that sow's own median over her first --baseline-days days, because sows differ
far more from each other than a sow differs from herself.

Usage:
    python estrus/sow_days.py outputs/gt_feb06/predictions.csv outputs/vulva/sow_day_truth.csv
    python estrus/sow_days.py predictions.csv sow_day_truth.csv --vulva outputs/vulva/vulva_daily.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta

NIGHT_START, NIGHT_END = 19, 8
MAX_GAP_MIN = 30          # longer gaps between frames do not count as a transition
POSTURES = ("standing", "sitting", "lying")
POSTURE_FEATURES = ["night_transitions", "night_transitions_per_h", "night_hours",
                    "night_standing_frac", "night_sitting_frac", "night_lying_frac",
                    "day_standing_frac", "day_lying_frac"]
VULVA_FEATURES = ["vulva_index", "width_mm", "length_mm", "volume_mm3"]


def parse_ts(ts):
    return datetime.strptime(ts[:17], "%Y%m%d-%H-%M-%S")


def night_key(dt):
    """Night 19:00 on D-1 .. 08:00 on D belongs to D; daytime frames return None."""
    if dt.hour >= NIGHT_START:
        return (dt + timedelta(days=1)).date()
    if dt.hour < NIGHT_END:
        return dt.date()
    return None


def load_posture(path):
    by_pig = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if (r.get("id_status") or "ok") not in ("ok", "", "corrected", "renamed"):
                continue
            name = (r.get("prediction_name") or "").strip()
            if name not in POSTURES:
                continue
            ts = r.get("pig_timestamp") or r.get("timestamp_folder")
            by_pig[int(float(r["pig_id"]))].append((parse_ts(ts), name))
    for v in by_pig.values():
        v.sort()
    return by_pig


def posture_features(frames):
    """frames: sorted (datetime, posture) for one sow -> {date: features}."""
    nights, days = defaultdict(list), defaultdict(list)
    for dt, p in frames:
        k = night_key(dt)
        if k is None:
            days[dt.date()].append((dt, p))
        else:
            nights[k].append((dt, p))
    out = defaultdict(dict)
    for d, fr in nights.items():
        trans, covered = 0, 0.0
        for (t0, p0), (t1, p1) in zip(fr, fr[1:]):
            gap = (t1 - t0).total_seconds() / 60
            if gap <= MAX_GAP_MIN:
                covered += gap
                trans += p0 != p1
        hours = covered / 60
        rec = out[d]
        rec["night_frames"] = len(fr)
        rec["night_hours"] = round(hours, 2)
        rec["night_transitions"] = trans
        rec["night_transitions_per_h"] = round(trans / hours, 3) if hours > 0 else None
        for p in POSTURES:
            rec[f"night_{p}_frac"] = round(sum(x == p for _, x in fr) / len(fr), 3)
    for d, fr in days.items():
        rec = out[d]
        rec["day_frames"] = len(fr)
        rec["day_standing_frac"] = round(sum(x == "standing" for _, x in fr) / len(fr), 3)
        rec["day_lying_frac"] = round(sum(x == "lying" for _, x in fr) / len(fr), 3)
    return out


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build(posture_by_pig, truth_rows, vulva_rows=None, baseline_days=2, min_night_hours=6.0):
    vul = {}
    for r in vulva_rows or []:
        vul[(int(float(r["pig_id"])), r["date"])] = r
    truth = defaultdict(dict)
    for r in truth_rows:
        truth[int(float(r["pig_id"]))][r["date"]] = r
    rows = []
    for pid, days in sorted(truth.items()):
        feats = posture_features(posture_by_pig.get(pid, []))
        for d_str, t in sorted(days.items()):
            d = date.fromisoformat(d_str)
            rec = {"pig_id": pid, "date": d_str, "dfw": int(float(t["day"])),
                   "heat": int(float(t.get("heat") or 0)), "action": t.get("action", "")}
            f = feats.get(d, {})
            for k in POSTURE_FEATURES + ["night_frames", "day_frames"]:
                rec[k] = f.get(k)
            if rec["night_hours"] is not None and rec["night_hours"] < min_night_hours:
                for k in [k for k in POSTURE_FEATURES if k.startswith("night_")]:
                    rec[k] = None      # too little of the night was seen
            v = vul.get((pid, d_str))
            for k in VULVA_FEATURES:
                rec[k] = _f(v.get(k)) if v else None
            rows.append(rec)
    # change from each sow's own first days
    by = defaultdict(list)
    for r in rows:
        by[r["pig_id"]].append(r)
    for rs in by.values():
        rs.sort(key=lambda r: r["date"])
        for k in POSTURE_FEATURES + VULVA_FEATURES:
            base = [r[k] for r in rs if r[k] is not None][:baseline_days]
            b = statistics.median(base) if base else None
            for r in rs:
                r[k + "_delta"] = round(r[k] - b, 4) if (b is not None and r[k] is not None) else None
    # first heat day per sow, and days relative to it
    for rs in by.values():
        heat_days = [r["dfw"] for r in rs if r["heat"]]
        onset = min(heat_days) if heat_days else None
        for r in rs:
            r["onset_dfw"] = onset
            r["days_to_onset"] = (r["dfw"] - onset) if onset is not None else None
    return rows


def fields_for(rows):
    first = ["pig_id", "date", "dfw", "heat", "onset_dfw", "days_to_onset", "action"]
    return first + [k for k in rows[0] if k not in first] if rows else first


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", help="posture predictions.csv")
    ap.add_argument("truth", help="sow_day_truth.csv from vulva/truth.py")
    ap.add_argument("--vulva", default="", help="vulva_daily.csv from vulva/run_vulva.py")
    ap.add_argument("--baseline-days", type=int, default=2)
    ap.add_argument("--min-night-hours", type=float, default=6.0)
    ap.add_argument("--out", default=os.path.join("outputs", "estrus", "sow_day_features.csv"))
    args = ap.parse_args()

    with open(args.truth, newline="", encoding="utf-8-sig") as f:
        truth = list(csv.DictReader(f))
    vulva = None
    if args.vulva:
        with open(args.vulva, newline="", encoding="utf-8-sig") as f:
            vulva = list(csv.DictReader(f))
    rows = build(load_posture(args.predictions), truth, vulva, args.baseline_days, args.min_night_hours)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields_for(rows))
        w.writeheader()
        w.writerows(rows)
    with_night = sum(r["night_transitions"] is not None for r in rows)
    print(f"{len(rows)} sow-days ({sum(r['heat'] for r in rows)} in heat); "
          f"{with_night} with a usable night of posture data -> {args.out}")


if __name__ == "__main__":
    main()
