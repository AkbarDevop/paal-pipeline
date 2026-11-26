#!/usr/bin/env python3
r"""What is in a labelled set: frames by day, time of day, stall and source, so new labels can go where the set
is thin (days, hours, stalls it has few or none of).

    python vulva\data_summary.py C:\Users\aakdbx\vulva_dataset_v1\labels.csv               # every split
    python vulva\data_summary.py C:\Users\aakdbx\vulva_dataset_v1\labels.csv --split train

Works on dataset tables (build_dataset.py: date, hour, animal_id, source columns) and on plain label sets
(date and hour taken from pig_timestamp / timestamp_folder, YYYYMMDD-HH-MM-SS).
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict

PARTS = (("night 18-06", lambda h: h >= 18 or h < 6), ("morning 06-11", lambda h: 6 <= h < 11),
         ("midday 11-14", lambda h: 11 <= h < 14), ("afternoon 14-18", lambda h: 14 <= h < 18))


def date_hour(r):
    """-> ('YYYYMMDD', hour) from the date / hour columns, else from the frame's timestamp."""
    d, h = r.get("date", ""), r.get("hour", "")
    if d and str(h) != "":
        return d, int(float(h))
    ts = r.get("pig_timestamp") or r.get("timestamp_folder") or ""
    try:
        return ts[:8], int(ts[9:11])
    except ValueError:
        return ts[:8], -1


def summarise(rows):
    lines = []
    vis = [r for r in rows if r.get("label") == "present"]
    lines.append(f"  {len(rows)} frames, {len(vis)} with a visible vulva, {len(rows) - len(vis)} without")
    by_day, by_day_vis = Counter(), Counter()
    hours = Counter()
    for r in rows:
        d, h = date_hour(r)
        by_day[d] += 1
        by_day_vis[d] += r.get("label") == "present"
        hours[h] += 1
    days = sorted(by_day)
    lines.append(f"  {len(days)} days, {days[0] if days else '-'} to {days[-1] if days else '-'}")
    months = defaultdict(list)
    for d in days:
        months[d[:6]].append(d)
    for m, ds in sorted(months.items()):
        lines.append(f"    {m[:4]}-{m[4:]}: {sum(by_day[d] for d in ds)} frames on {len(ds)} days: "
                     + ", ".join(f"{d[6:]}:{by_day[d]}" for d in ds))
    parts = [(name, sum(n for h, n in hours.items() if h >= 0 and test(h))) for name, test in PARTS]
    lines.append("  time of day: " + ", ".join(f"{name} {n} ({100 * n / max(len(rows), 1):.0f}%)" for name, n in parts))
    lines.append("  by hour: " + " ".join(f"{h:02d}:{hours[h]}" for h in range(24) if hours[h]))
    stalls = Counter(r.get("animal_id") or f"s{int(float(r.get('pig_id') or -1)):02d}" for r in rows)
    lines.append(f"  {len(stalls)} stalls (batch + stall): " + ", ".join(f"{k} {v}" for k, v in sorted(stalls.items())))
    src = Counter(r.get("source", "") for r in rows if r.get("source"))
    if src:
        lines.append("  from: " + ", ".join(f"{k} {v}" for k, v in src.most_common()))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", help="labels.csv (dataset table or label set)")
    ap.add_argument("--split", default="", help="only this split (default: each split in turn)")
    args = ap.parse_args()
    with open(args.labels, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("label") in ("present", "absent")]
    groups = defaultdict(list)
    for r in rows:
        if not args.split or r.get("split", "") == args.split:
            groups[r.get("split", "") or "all"].append(r)
    out = []
    for split in sorted(groups):
        out.append(f"\n{split}")
        out.extend(summarise(groups[split]))
    print("\n".join(out))
    return out


if __name__ == "__main__":
    main()
