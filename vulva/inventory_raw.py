#!/usr/bin/env python3
r"""What raw data do we have? Every robot pass (timestamp folder) under the raw roots, counted
by month and by day: passes, sow images, night passes (18:00-06:00), first and last pass, and
the days without any pass. Read-only. Used to plan test sets (a later batch of sows, night
frames) and to see camera gaps.

    python vulva\inventory_raw.py $RAW
    python vulva\inventory_raw.py "E:\Pig_data\Pictures_OAK;$RAW" --quick     # passes only, no file counts

Writes outputs\raw_inventory_days.csv (one row per day) unless --out is given.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IR = re.compile(r"^pig(\d+)_ir_vis_.*\.(jpg|png)$", re.IGNORECASE)


def count_frames(path):
    """-> (sow images, highest stall number) in one timestamp folder."""
    n, top = 0, -1
    try:
        for e in os.scandir(path):
            m = IR.match(e.name)
            if m:
                n += 1
                top = max(top, int(m.group(1)))
    except OSError:
        return 0, -1
    return n, top


def inventory(roots, quick=False):
    from select_frames import index_folders
    folders = index_folders(roots)
    days = defaultdict(lambda: {"passes": 0, "night": 0, "frames": 0, "first": "99", "last": "00", "max_stall": -1})
    for k, (name, path) in enumerate(sorted(folders.items())):
        d, hh = name[:8], int(name[9:11])
        r = days[d]
        r["passes"] += 1
        r["night"] += hh >= 18 or hh < 6
        r["first"], r["last"] = min(r["first"], name[9:]), max(r["last"], name[9:])
        if not quick:
            n, top = count_frames(path)
            r["frames"] += n
            r["max_stall"] = max(r["max_stall"], top)
        if k and k % 500 == 0:
            print(f"  {k} of {len(folders)} passes read", flush=True)
    return days


def gaps(day_keys):
    if not day_keys:
        return []
    ds = [date(int(d[:4]), int(d[4:6]), int(d[6:8])) for d in sorted(day_keys)]
    have, out, cur = set(ds), [], ds[0]
    while cur <= ds[-1]:
        if cur not in have:
            start = cur
            while cur + timedelta(days=1) not in have and cur + timedelta(days=1) <= ds[-1]:
                cur += timedelta(days=1)
            out.append((start, cur))
        cur += timedelta(days=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", help="';'-separated raw folders (timestamp folders directly or one level down)")
    ap.add_argument("--quick", action="store_true", help="count passes only (no file listing; fast on a network share)")
    ap.add_argument("--out", default=os.path.join("outputs", "raw_inventory_days.csv"))
    args = ap.parse_args()
    roots = [r for r in args.roots.split(";") if r.strip()]
    days = inventory(roots, args.quick)
    if not days:
        sys.exit(f"No timestamp folders (YYYYMMDD-HH-MM-SS) under {roots}")
    months = defaultdict(lambda: {"days": 0, "passes": 0, "night": 0, "frames": 0})
    for d, r in days.items():
        m = months[d[:6]]
        m["days"] += 1
        for k in ("passes", "night", "frames"):
            m[k] += r[k]
    first, last = min(days), max(days)
    print(f"{sum(r['passes'] for r in days.values())} robot passes on {len(days)} days, "
          f"{first[:4]}-{first[4:6]}-{first[6:]} to {last[:4]}-{last[4:6]}-{last[6:]}")
    print("month     days  passes  night passes" + ("" if args.quick else "  sow images"))
    for m in sorted(months):
        r = months[m]
        print(f"{m[:4]}-{m[4:]}   {r['days']:4d}  {r['passes']:6d}  {r['night']:12d}"
              + ("" if args.quick else f"  {r['frames']:10d}"))
    gl = gaps(days.keys())
    if gl:
        print("days without any pass: " + ", ".join(f"{a:%m-%d}" if a == b else f"{a:%m-%d} to {b:%m-%d}" for a, b in gl))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "passes", "night_passes", "sow_images", "first_pass", "last_pass", "highest_stall"])
        for d in sorted(days):
            r = days[d]
            w.writerow([d, r["passes"], r["night"], r["frames"] if not args.quick else "", r["first"], r["last"],
                        r["max_stall"] if not args.quick else ""])
    print(f"per-day table -> {args.out}")


if __name__ == "__main__":
    main()
