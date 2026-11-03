#!/usr/bin/env python3
r"""Which outline size is right? Camera widths and lengths against Lucas's caliper values.

Fresh hand outlines on the blind test are about 15% bigger than the reviewed training outlines
and the models drawn from them. The calipers decide. For every source this takes the median
width and length per sow per day, joins it with the caliper row of that sow-day (truth.py) and
reports how far off it is (median ratio camera / caliper, mean difference, mean absolute
difference) and how well it follows the calipers (r across sow-days, and r within a sow, i.e.
of each sow's day-to-day changes, which is what heat detection needs).

Sources (any number):
  eval_frames.csv  from eval_seg.py: gives two sources, the hand outlines and the model's kept outlines
  vulva_frames.csv from run_vulva.py: the kept, well-measured frames (the ones the daily values use)

    python vulva/size_vs_caliper.py outputs/vulva/sow_day_truth.csv ^
        outputs/eval_r8_feb/eval_frames.csv outputs/vulva_r8_long/vulva_frames.csv

Writes size_vs_caliper.txt and size_vs_caliper.csv (one row per source and sow-day) next to the
first source unless --out-dir is given.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from daily import frame_date, is_good, pearson  # noqa: E402


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_truth(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        out = {}
        for r in csv.DictReader(f):
            w, l = _f(r.get("vulva_width_mm")), _f(r.get("vulva_length_mm"))
            if w and l:
                out[(int(float(r["pig_id"])), str(r["date"]))] = (w, l)
    return out


def sources_from(path):
    """[(name, [(pig, date, width, length), ...])] from an eval or run CSV."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    tag = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if rows and "human_width_mm" in rows[0]:
        hand, model = [], []
        for r in rows:
            if r.get("label") != "present":
                continue
            key = (int(float(r["pig_id"])), frame_date(r))
            if r.get("human_status") == "ok":
                hand.append((*key, _f(r["human_width_mm"]), _f(r["human_length_mm"])))
            if r.get("model_visible") == "1" and r.get("model_status") == "ok":
                model.append((*key, _f(r["model_width_mm"]), _f(r["model_length_mm"])))
        return [(f"hand outlines ({tag})", hand, 1), (f"model, kept ({tag})", model, 1)]
    run = [(int(float(r["pig_id"])), frame_date(r), _f(r["width_mm"]), _f(r["length_mm"]))
           for r in rows if str(r.get("vulva_visible", "1")) == "1" and is_good(r)]
    return [(f"run ({tag})", run, 3)]


def compare(frames, truth, min_frames=1):
    groups = defaultdict(list)
    for pig, day, w, l in frames:
        if w and l:
            groups[(pig, day)].append((w, l))
    joined = []
    for key, vals in sorted(groups.items()):
        if len(vals) < min_frames or key not in truth:
            continue
        cw, cl = truth[key]
        joined.append({"pig_id": key[0], "date": key[1], "n_frames": len(vals),
                       "width_mm": statistics.median(v[0] for v in vals),
                       "length_mm": statistics.median(v[1] for v in vals),
                       "caliper_width_mm": cw, "caliper_length_mm": cl})
    stats = {"sow_days": len(joined), "sows": len({j["pig_id"] for j in joined})}
    for k in ("width_mm", "length_mm"):
        cam = [j[k] for j in joined]
        cal = [j["caliper_" + k] for j in joined]
        if len(cam) < 2:
            continue
        stats[k] = {"ratio": statistics.median(a / b for a, b in zip(cam, cal)),
                    "bias": statistics.mean(a - b for a, b in zip(cam, cal)),
                    "mae": statistics.mean(abs(a - b) for a, b in zip(cam, cal)),
                    "r": pearson(cam, cal), "within_r": within_sow_r(joined, k)}
    return joined, stats


def within_sow_r(joined, k):
    """r of each sow's day-to-day changes: both series minus that sow's own mean."""
    by = defaultdict(list)
    for j in joined:
        by[j["pig_id"]].append(j)
    a, b = [], []
    for js in by.values():
        if len(js) < 3:
            continue
        ma = statistics.mean(j[k] for j in js)
        mb = statistics.mean(j["caliper_" + k] for j in js)
        a += [j[k] - ma for j in js]
        b += [j["caliper_" + k] - mb for j in js]
    return pearson(a, b) if len(a) >= 6 else None


def fmt_r(r):
    return "n/a" if r is None else f"{r:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("truth", help="sow_day_truth.csv from truth.py")
    ap.add_argument("sources", nargs="+", help="eval_frames.csv and/or vulva_frames.csv files")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    truth = read_truth(args.truth)
    lines = [f"Calipers: {len(truth)} sow-days with width and length ({args.truth})"]
    table = []
    for path in args.sources:
        for name, frames, min_frames in sources_from(path):
            joined, s = compare(frames, truth, min_frames)
            lines.append(f"{name}: {s['sow_days']} sow-days of {s['sows']} sows matched "
                         f"(at least {min_frames} frame{'s' if min_frames > 1 else ''} each)")
            for k, label in (("width_mm", "width"), ("length_mm", "length")):
                if k in s:
                    v = s[k]
                    lines.append(f"  {label}: camera / caliper {v['ratio']:.2f} (median), difference "
                                 f"{v['bias']:+.1f} mm (mean), off by {v['mae']:.1f} mm on average; "
                                 f"r {fmt_r(v['r'])} across sow-days, {fmt_r(v['within_r'])} within a sow")
            table += [dict(j, source=name) for j in joined]
    print("\n".join(lines))
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.sources[0]))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "size_vs_caliper.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    fields = ["source", "pig_id", "date", "n_frames", "width_mm", "caliper_width_mm", "length_mm", "caliper_length_mm"]
    with open(os.path.join(out_dir, "size_vs_caliper.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table)


if __name__ == "__main__":
    main()
