"""Score predictions against Lucas's time-stamped posture ground truth.

Lucas's "sow position.xlsx" has one sheet per day (sheet "26" = Feb 6,
"210" = Feb 10), a time column and one column per pig with codes
0 = lying, 1 = sitting, 2 = standing, recorded about every 10 minutes.
The OAK frames for those days are in OAK_0206_0211 on the room-127 PC
(found by Dr. Zhou on Apr 19), so this is the first independent test of the
model: different days from the training labels (Jan 23 - Feb 3) and labelled
by the animal-science side, not by us.

Steps:
  1. python posture/run_inference.py <OAK_0206_0211 folder> --output-dir outputs/gt_feb06 --smooth 3
  2. python posture/eval_ground_truth.py --gt "sow position.xlsx" --pred outputs/gt_feb06/predictions.csv

If the sheet layout differs, run with --inspect first: it prints the first
rows of every sheet so the column/row options can be set.

Matching: each ground-truth cell (time, pig) is paired with the prediction
for the same pig whose timestamp is closest, within --tolerance minutes.
Reports accuracy / macro-F1 / confusion for raw and smoothed predictions and
compares nightly transition counts (ground truth vs predicted).
"""

import argparse
import csv
import os
import re
import sys
import pathlib
from collections import defaultdict
from datetime import datetime, time as dtime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import OUTPUT_DIR

NAMES = ["standing", "sitting", "lying"]


def parse_codes(spec):
    out = {}
    for part in spec.split(","):
        k, v = part.split(":")
        out[str(k).strip()] = v.strip()
    return out


def sheet_date(name, year, explicit=None):
    """'26' -> Feb 6, '210' -> Feb 10, '2-6' / '02_06' also accepted."""
    if explicit:
        return datetime.strptime(explicit, "%Y-%m-%d").date()
    digits = re.sub(r"\D", "", name)
    if not digits:
        return None
    month, day = int(digits[0]), int(digits[1:]) if len(digits) > 1 else None
    if day is None:
        return None
    try:
        return datetime(year, month, day).date()
    except ValueError:
        return None


def to_time(v):
    if isinstance(v, datetime):
        return v.time()
    if isinstance(v, dtime):
        return v
    if isinstance(v, (int, float)) and 0 <= v < 1:      # Excel day fraction
        secs = round(v * 86400)
        return dtime(secs // 3600 % 24, secs // 60 % 60)
    if isinstance(v, str):
        for fmt in ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p", "%H%M"):
            try:
                return datetime.strptime(v.strip(), fmt).time()
            except ValueError:
                continue
    return None


def pig_from_header(h):
    if h is None:
        return None
    m = re.fullmatch(r"\s*(?:pig|p|stall|s)?\s*#?\s*(\d{1,2})\s*", str(h), re.I)
    if m and 0 <= int(m.group(1)) < 40:
        return int(m.group(1))
    if isinstance(h, (int, float)) and float(h).is_integer() and 0 <= int(h) < 40:
        return int(h)
    return None


def find_layout(rows):
    """Header row = first row with >= 10 pig-like headers; time column = most time-like."""
    for i, row in enumerate(rows[:30]):
        pigs = {j: pig_from_header(v) for j, v in enumerate(row)}
        pigs = {j: p for j, p in pigs.items() if p is not None}
        if len(pigs) >= 10:
            body = rows[i + 1:i + 40]
            best, best_n = None, 0
            for j in range(len(row)):
                if j in pigs:
                    continue
                n = sum(1 for r in body if j < len(r) and to_time(r[j]) is not None)
                if n > best_n:
                    best, best_n = j, n
            return i, best, pigs
    return None, None, {}


def read_ground_truth(path, codes, year, stall_offset=0, sheets=None):
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    out = []
    for name in wb.sheetnames:
        if sheets and name not in sheets:
            continue
        day = sheet_date(name, year)
        rows = [list(r) for r in wb[name].iter_rows(values_only=True)]
        hdr, tcol, pigs = find_layout(rows)
        if day is None or hdr is None or tcol is None:
            print(f"  sheet '{name}': skipped (could not find date/time/pig columns; try --inspect)")
            continue
        n = 0
        for r in rows[hdr + 1:]:
            t = to_time(r[tcol]) if tcol < len(r) else None
            if t is None:
                continue
            ts = datetime.combine(day, t)
            for j, pig in pigs.items():
                v = r[j] if j < len(r) else None
                if v is None or str(v).strip() == "":
                    continue
                key = str(int(v)) if isinstance(v, (int, float)) and float(v).is_integer() else str(v).strip()
                label = codes.get(key)
                if label:
                    out.append({"time": ts, "pig_id": pig - stall_offset, "label": label})
                    n += 1
        print(f"  sheet '{name}' -> {day}: {n} labels, pigs {sorted(pigs.values())[:3]}..{sorted(pigs.values())[-1]}")
    return out


def read_predictions(path):
    by_pig = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                t = datetime.strptime(r["pig_timestamp"], "%Y%m%d-%H-%M-%S")
            except ValueError:
                continue
            by_pig[int(r["pig_id"])].append((t, r))
    for v in by_pig.values():
        v.sort(key=lambda x: x[0])
    return by_pig


def nearest(seq, t, tol):
    import bisect
    times = [x[0] for x in seq]
    i = bisect.bisect_left(times, t)
    best = None
    for k in (i - 1, i):
        if 0 <= k < len(seq):
            d = abs((seq[k][0] - t).total_seconds()) / 60.0
            if d <= tol and (best is None or d < best[0]):
                best = (d, seq[k][1])
    return best[1] if best else None


def metrics(pairs):
    idx = {n: i for i, n in enumerate(NAMES)}
    cm = [[0] * 3 for _ in range(3)]
    for t, p in pairs:
        cm[idx[t]][idx[p]] += 1
    n = sum(map(sum, cm))
    acc = sum(cm[i][i] for i in range(3)) / n if n else 0.0
    f1s = []
    for c in range(3):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(3)) - tp
        fn = sum(cm[c]) - tp
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"n": n, "accuracy": acc, "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0, "confusion": cm}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="Ground-truth workbook (sow position.xlsx)")
    ap.add_argument("--pred", default=os.path.join(OUTPUT_DIR, "predictions.csv"))
    ap.add_argument("--codes", default="0:lying,1:sitting,2:standing",
                    help="Ground-truth code mapping (default from Lucas's sheet)")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--tolerance", type=float, default=6.0, help="Max minutes between GT and frame")
    ap.add_argument("--stall-offset", type=int, default=0,
                    help="Subtract from GT pig numbers if the sheet counts stalls from 1")
    ap.add_argument("--sheets", nargs="*", default=None)
    ap.add_argument("--inspect", action="store_true", help="Print the first rows of each sheet and exit")
    ap.add_argument("--out", default=os.path.join(OUTPUT_DIR, "ground_truth_eval.csv"))
    args = ap.parse_args()

    if args.inspect:
        import openpyxl
        wb = openpyxl.load_workbook(args.gt, data_only=True, read_only=True)
        for name in wb.sheetnames:
            print(f"\n=== sheet {name!r} ===")
            for i, r in enumerate(wb[name].iter_rows(values_only=True)):
                print(i, list(r)[:24])
                if i >= 8:
                    break
        return

    gt = read_ground_truth(args.gt, parse_codes(args.codes), args.year, args.stall_offset, args.sheets)
    preds = read_predictions(args.pred)
    has_smooth = any("smoothed_prediction_name" in r for seq in preds.values() for _, r in seq[:1])

    rows, raw_pairs, sm_pairs = [], [], []
    per_pig = defaultdict(lambda: [0, 0])
    unmatched = 0
    for g in gt:
        p = nearest(preds.get(g["pig_id"], []), g["time"], args.tolerance)
        if p is None:
            unmatched += 1
            continue
        raw = p["prediction_name"]
        sm = p.get("smoothed_prediction_name") or raw
        raw_pairs.append((g["label"], raw))
        sm_pairs.append((g["label"], sm))
        per_pig[g["pig_id"]][0] += int(raw == g["label"])
        per_pig[g["pig_id"]][1] += 1
        rows.append({"time": g["time"].strftime("%Y-%m-%d %H:%M"), "pig_id": g["pig_id"], "ground_truth": g["label"],
                     "prediction": raw, "smoothed": sm, "frame": p["pig_timestamp"],
                     "confidence": p.get("confidence", "")})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["time"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nGround-truth cells: {len(gt)} | matched to a frame within {args.tolerance:.0f} min: {len(rows)} "
          f"| unmatched: {unmatched}")
    for tag, pairs in (("raw", raw_pairs),) + ((("smoothed", sm_pairs),) if has_smooth else ()):
        m = metrics(pairs)
        print(f"  {tag:<9} accuracy {m['accuracy']*100:.2f}%  macro-F1 {m['macro_f1']*100:.2f}%  "
              f"confusion (rows=true {NAMES}): {m['confusion']}")
    print("  per-pig accuracy: " + ", ".join(f"{p}:{c / n * 100:.0f}%" for p, (c, n) in sorted(per_pig.items())))

    # transitions on the matched GT grid (same instants for truth and prediction)
    by = defaultdict(list)
    for r in rows:
        by[r["pig_id"]].append(r)
    tr = {"ground_truth": 0, "prediction": 0, "smoothed": 0}
    for seq in by.values():
        seq.sort(key=lambda r: r["time"])
        for a, b in zip(seq, seq[1:]):
            gap = (datetime.strptime(b["time"], "%Y-%m-%d %H:%M") -
                   datetime.strptime(a["time"], "%Y-%m-%d %H:%M")).total_seconds() / 60
            if gap > 30:
                continue
            for k in tr:
                tr[k] += int(a[k] != b[k])
    print(f"  transitions on the GT grid: truth {tr['ground_truth']} | predicted {tr['prediction']}"
          + (f" | smoothed {tr['smoothed']}" if has_smooth else ""))
    print(f"\nRow-level comparison: {args.out}")
    if per_pig and min(c / n for c, n in per_pig.values()) < 0.6:
        print("  Note: a pig far below the others usually means a camera-ID / stall shift on those days, "
              "not a model error — check data_quality.csv for those folders.")


if __name__ == "__main__":
    main()
