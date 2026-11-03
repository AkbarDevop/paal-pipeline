#!/usr/bin/env python3
r"""Pick the frames worth labelling next from a run_vulva.py run (active learning).

Random frames are mostly easy ones the model already gets right. This picks where the model
struggles, spread over as many sows and days as possible:

  unsure    the model drew an outline but dropped it as low confidence (tails, far sows, real misses)
  borderline kept, but only just above the model's confidence cut
  nothing   the model drew nothing (were there visible vulvas it missed?)
  sure      kept with high confidence (a few, so sizes are labelled on easy frames too)

Frames from the blind tests are never picked: --hold-out-dates drops whole days (the Feb 12–15
test, a new test period) and --hold-out drops sows on a date range (the January test sows).
Frames already in any --exclude-labels CSV are skipped.

Writes a predictions-style CSV that export_label_set.py copies out like any other set:

    python vulva/pick_hard_frames.py outputs\vulva_r8_long\vulva_frames.csv --model models\vulva_seg_r8_ir.pth ^
        --hold-out-dates 20260212-20260215,20260302-20260312 ^
        --hold-out 6,7,8,10,11,15,16,18@20260123-20260205 --out outputs\r9_pick.csv
    python vulva/export_label_set.py outputs\r9_pick.csv C:\Users\aakdbx\vulva_r9_set --per-day 0 --windows 00:00-23:59 --rgb
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BUCKETS = ("unsure", "borderline", "nothing", "sure")


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_ranges(spec):
    """'20260212-20260215,20260301' -> [(lo, hi), ...] of YYYYMMDD strings."""
    out = []
    for part in (p.strip() for p in (spec or "").split(",")):
        if part:
            lo, _, hi = part.partition("-")
            out.append((lo, hi or lo))
    return out


def parse_holdout(spec):
    """'6,7,8@20260123-20260205;3@20260301-20260302' -> [({6, 7, 8}, lo, hi), ...]"""
    out = []
    for part in (p.strip() for p in (spec or "").split(";")):
        if part:
            sows, _, dates = part.partition("@")
            lo, _, hi = (dates or "00000000-99999999").partition("-")
            out.append(({int(s) for s in sows.split(",") if s.strip()}, lo, hi or lo))
    return out


def held_out(row, dates, sows):
    day = str(row.get("pig_timestamp") or row.get("timestamp_folder") or "")[:8]
    if any(lo <= day <= hi for lo, hi in dates):
        return True
    pig = int(_f(row.get("pig_id"), -1))
    return any(pig in s and lo <= day <= hi for s, lo, hi in sows)


def labelled_keys(paths):
    keys = set()
    for p in paths:
        with open(p, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                keys.add((str(int(_f(r.get("pig_id"), -1))), str(r.get("pig_timestamp", ""))))
    return keys


def bucket(row, cut, margin=0.006, sure=0.999):
    status, conf = row.get("measure_status", ""), _f(row.get("vulva_conf"))
    if status == "low_confidence":
        return "unsure"
    if status == "not_visible":
        return "nothing"
    if str(row.get("vulva_visible")) == "1":
        if conf < cut + margin:
            return "borderline"
        if conf >= sure:
            return "sure"
    return None


def spread_pick(rows, n, rng):
    """n rows spread round-robin over (sow, day) groups, random inside each group."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["pig_id"], str(r["pig_timestamp"])[:8])].append(r)
    keys = list(groups)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(groups[k])
    out = []
    while len(out) < n and keys:
        nxt = []
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
            if groups[k]:
                nxt.append(k)
        keys = nxt
    return out


def pick(rows, cut, counts, rng, dates=(), sows=(), skip=frozenset()):
    pools = {b: [] for b in BUCKETS}
    for r in rows:
        if held_out(r, dates, sows) or (str(int(_f(r.get("pig_id"), -1))), str(r.get("pig_timestamp", ""))) in skip:
            continue
        b = bucket(r, cut)
        if b:
            pools[b].append(r)
    picked = []
    for b in BUCKETS:
        for r in spread_pick(pools[b], counts.get(b, 0), rng):
            picked.append(dict(r, bucket=b))
    return picked, {b: len(v) for b, v in pools.items()}


def summary(rows):
    lines = []
    st = Counter(r.get("measure_status", "") for r in rows)
    lines.append("Run: " + ", ".join(f"{k or 'blank'} {v}" for k, v in st.most_common()))
    by_month = defaultdict(Counter)
    for r in rows:
        by_month[str(r.get("pig_timestamp", ""))[:6]][r.get("measure_status", "")] += 1
    for m, c in sorted(by_month.items()):
        tot = sum(c.values())
        lines.append(f"  {m[:4]}-{m[4:6]}: {tot} standing frames, kept {c['ok'] / max(tot, 1):.0%}, "
                     f"dropped as unsure {c['low_confidence'] / max(tot, 1):.0%}, "
                     f"nothing drawn {c['not_visible'] / max(tot, 1):.0%}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames_csv", help="vulva_frames.csv from run_vulva.py")
    ap.add_argument("--model", required=True, help="checkpoint the run used (for its confidence cut)")
    ap.add_argument("--unsure", type=int, default=70)
    ap.add_argument("--borderline", type=int, default=30)
    ap.add_argument("--nothing", type=int, default=30)
    ap.add_argument("--sure", type=int, default=20)
    ap.add_argument("--hold-out-dates", default="", help="YYYYMMDD-YYYYMMDD ranges never picked (test days)")
    ap.add_argument("--hold-out", default="", help="sows@dates never picked, e.g. 6,7,8@20260123-20260205")
    ap.add_argument("--exclude-labels", nargs="*", default=[], help="labels CSVs whose frames are already done")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("outputs", "hard_frames.csv"))
    args = ap.parse_args()

    import torch
    cut = float(torch.load(args.model, map_location="cpu", weights_only=False).get("presence_threshold", 0.0))
    with open(args.frames_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    counts = {"unsure": args.unsure, "borderline": args.borderline, "nothing": args.nothing, "sure": args.sure}
    picked, pools = pick(rows, cut, counts, random.Random(args.seed), parse_ranges(args.hold_out_dates),
                         parse_holdout(args.hold_out), labelled_keys(args.exclude_labels))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fields = ["timestamp_folder", "pig_id", "pig_timestamp", "prediction_name", "confidence", "image_path",
              "bucket", "vulva_conf", "measure_status"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in picked:
            w.writerow(dict(r, prediction_name="standing", confidence=r.get("posture_confidence") or 1,
                            image_path=r["ir_path"]))
    print("\n".join(summary(rows)))
    print(f"Confidence cut {cut}. Candidates after hold-outs: " + ", ".join(f"{b} {pools[b]}" for b in BUCKETS))
    got = Counter(r["bucket"] for r in picked)
    print(f"Picked {len(picked)} frames (" + ", ".join(f"{b} {got[b]}" for b in BUCKETS) + f") from "
          f"{len({r['pig_id'] for r in picked})} sows and {len({str(r['pig_timestamp'])[:8] for r in picked})} days "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
