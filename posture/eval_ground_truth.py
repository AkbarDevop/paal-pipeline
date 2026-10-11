"""Score predictions against Lucas's time-stamped posture ground truth.

Lucas's "sow position.xlsx" has one sheet per day (sheet "26" = Feb 6,
"210" = Feb 10, up to "215"), one row per robot pass (~5.5 min) with the time typed
on about every other row, one column per pig with codes 0 = lying, 1 = sitting,
2 = standing ('X' = empty stall), and his per-sow totals under each block. Some
days also have 'completo' / 'completo alternado' sheets (daytime too) and a
'Cópia' copy; all are merged into one label per pig per minute.
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
for the same pig whose timestamp is closest, within --tolerance minutes (default 3; his rows are ~5.5 min apart).
Reports accuracy / macro-F1 / confusion for raw and smoothed predictions and
compares nightly transition counts (ground truth vs predicted).

Presence check: Lucas recorded a sow in the stall for every ground-truth cell,
so a cell with no prediction is a miss. Using skipped_frames.csv and
predictions_excluded.csv from the same run, each miss is explained as
skipped by the depth gate, rejected by the presence CNN, unreadable, set aside
by the ID policy, or no capture at all, and broken down by the recorded
posture (a depth gate that drops lying sows shows up here). Misses are written
to ground_truth_misses.csv for visual review (posture/review_presence.py).
"""

import argparse
import csv
import os
import re
import sys
import pathlib
from collections import defaultdict
from datetime import datetime, timedelta, time as dtime

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
    """'26' -> Feb 6, '210' -> Feb 10; '2-6', '02_06', '2.10' also accepted."""
    if explicit:
        return datetime.strptime(explicit, "%Y-%m-%d").date()
    parts = [p for p in re.split(r"\D+", name) if p]
    if len(parts) >= 2:
        month, day = int(parts[0]), int(parts[1])
    elif len(parts) == 1 and len(parts[0]) >= 2:
        month, day = int(parts[0][0]), int(parts[0][1:])
    else:
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


SKIP_SHEET_RE = re.compile(r"c[oó]pia|copy", re.I)
ROW_MINUTES = 5.5          # one camera pass, used only when a time cannot be interpolated


def sheet_priority(name):
    """Several sheets can cover the same day. Prefer every-pass sheets ('210 completo', '210')
    over the every-other-pass 'completo alternado' ones."""
    n = name.lower()
    return 2 if "altern" in n else (0 if "complet" in n else 1)


SUMMARY_LABELS = {"standing", "sitting", "laying down", "lying", "lying down", "total"}


def _is_number(k):
    try:
        float(k)
        return True
    except (TypeError, ValueError):
        return False


def observation_rows(rows, hdr, tcol, pigs, codes):
    """Rows of the observation block. Empty rows and note rows ('Now sow 18', 'Sow 0 culled')
    are skipped; 'X' cells (empty stall) and other text cells are kept as non-codes. The block
    ends at the per-sow totals: a row with a number that is not a posture code, or the
    'standing' / 'sitting' / 'laying down' labels."""
    block = []
    for i in range(hdr + 1, len(rows)):
        r = rows[i]
        t_raw = r[tcol] if tcol < len(r) else None
        if isinstance(t_raw, str) and t_raw.strip().lower() in SUMMARY_LABELS:
            break
        keys = []
        for j in pigs:
            v = r[j] if j < len(r) else None
            if v is None or str(v).strip() == "":
                keys.append(None)
            elif isinstance(v, (int, float)) and float(v).is_integer():
                keys.append(str(int(v)))
            else:
                keys.append(str(v).strip())
        if any(k is not None and _is_number(k) and k not in codes for k in keys):
            break                                     # per-sow totals start here
        t = to_time(t_raw)
        if t is None and not any(k in codes for k in keys if k):
            continue                                  # empty or note row
        block.append((i, t, keys))
    return block


def assign_times(block, day):
    """Lucas typed the time on about every other row; rows between are interpolated from the
    recorded times around them (never across gaps over 30 min). Returns
    [(datetime, keys, source)] with source 'recorded' or 'interpolated'. A recorded time that
    does not fit between its recorded neighbours is treated as a typo and interpolated."""
    times = [datetime.combine(day, t) if t else None for _, t, _ in block]
    rec = [k for k, t in enumerate(times) if t is not None]
    bad = set()
    for a, k in enumerate(rec):
        p = times[rec[a - 1]] if a > 0 else None
        n = times[rec[a + 1]] if a + 1 < len(rec) else None
        t = times[k]
        if p is not None and n is not None:
            if p < n and not (p < t < n):
                bad.add(k)
        elif p is not None and t <= p:
            bad.add(k)
        elif n is not None and t >= n:
            bad.add(k)
    for k in bad:                       # a mistyped time is re-derived from its neighbours
        times[k] = None
    rec = [k for k in rec if k not in bad]
    out = []
    for k, (_, _, keys) in enumerate(block):
        t = times[k]
        if t is not None:
            out.append((t, keys, "recorded"))
            continue
        prev = max((j for j in rec if j < k), default=None)
        nxt = min((j for j in rec if j > k), default=None)
        if prev is not None and nxt is not None and (times[nxt] - times[prev]).total_seconds() <= 1800:
            t = times[prev] + (times[nxt] - times[prev]) * ((k - prev) / (nxt - prev))
        elif prev is not None and (k - prev) <= 2:
            t = times[prev] + timedelta(minutes=ROW_MINUTES * (k - prev))
        elif nxt is not None and (nxt - k) <= 2:
            t = times[nxt] - timedelta(minutes=ROW_MINUTES * (nxt - k))
        else:
            continue
        out.append((t.replace(second=0, microsecond=0), keys, "interpolated"))
    return out


def read_ground_truth(path, codes, year, stall_offset=0, sheets=None, verbose=True):
    """Lucas's sow position workbook -> [{time, pig_id, label, sheet, time_source}].

    Handles his layout: one row per robot pass (~5.5 min), time typed on about every other
    row, per-sow totals under the block, extra 'completo' / 'alternado' / 'Cópia' sheets for
    some days (copies skipped; overlapping sheets merged, one label per pig per minute)."""
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    names = [n for n in wb.sheetnames if (not sheets or n in sheets)]
    seen, out, conflicts = {}, [], 0
    for name in sorted(names, key=sheet_priority):
        if SKIP_SHEET_RE.search(name):
            if verbose:
                print(f"  sheet '{name}': skipped (copy of another sheet)")
            continue
        day = sheet_date(name, year)
        rows = [list(r) for r in wb[name].iter_rows(values_only=True)]
        hdr, tcol, pigs = find_layout(rows)
        if day is None or hdr is None or tcol is None:
            if verbose:
                print(f"  sheet '{name}': skipped (could not find date/time/pig columns; try --inspect)")
            continue
        block = observation_rows(rows, hdr, tcol, list(pigs), codes)
        n = dup = 0
        for ts, keys, source in assign_times(block, day):
            for (j, pig), key in zip(pigs.items(), keys):
                label = codes.get(key) if key else None
                if not label:
                    continue
                k = (ts, pig - stall_offset)
                if k in seen:
                    dup += 1
                    conflicts += int(seen[k] != label)
                    continue
                seen[k] = label
                out.append({"time": ts, "pig_id": pig - stall_offset, "label": label, "sheet": name,
                            "time_source": source})
                n += 1
        if verbose:
            print(f"  sheet '{name}' -> {day}: {len(block)} passes, {n} new labels"
                  + (f", {dup} already in another sheet" if dup else ""))
    if verbose and conflicts:
        print(f"  {conflicts} pig-times labelled differently in two sheets (first sheet kept)")
    out.sort(key=lambda g: (g["time"], g["pig_id"]))
    return out


def read_predictions(path, time_key="timestamp_folder"):
    """{pig: [(time, row)]}. time_key 'timestamp_folder' = start of the robot pass, which is
    what Lucas's row time refers to (one time per row for all 20 stalls); 'pig_timestamp' =
    the stall's own capture time, up to ~3 min later in the pass."""
    by_pig = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                t = datetime.strptime(r[time_key], "%Y%m%d-%H-%M-%S")
            except ValueError:
                continue
            by_pig[int(r["pig_id"])].append((t, r))
    for v in by_pig.values():
        v.sort(key=lambda x: x[0])
    return by_pig


def read_frames(path, include_only=True, time_key="timestamp_folder"):
    """skipped_frames.csv / predictions_excluded.csv -> {pig_id: [(time, row), ...]}"""
    by_pig = defaultdict(list)
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if include_only and str(r.get("include", "True")) not in ("True", "1", "true"):
                continue
            try:
                t = datetime.strptime(r[time_key], "%Y%m%d-%H-%M-%S")
                by_pig[int(r["pig_id"])].append((t, r))
            except (ValueError, KeyError):
                continue
    for v in by_pig.values():
        v.sort(key=lambda x: x[0])
    return by_pig


MISS_REASONS = ["depth", "detector", "corrupt", "id_policy", "no_frame"]


def pearson(x, y):
    if len(x) < 3:
        return float("nan")
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return sxy / (sx * sy) if sx and sy else float("nan")


def class_line(tag, pairs):
    m = metrics(pairs)
    cm = m["confusion"]
    rec = [cm[i][i] / max(1, sum(cm[i])) * 100 for i in range(3)]
    pre = [cm[i][i] / max(1, sum(cm[j][i] for j in range(3))) * 100 for i in range(3)]
    print(f"  {tag:<20} n={m['n']:6d}  accuracy {m['accuracy']*100:5.1f}%  macro-F1 {m['macro_f1']*100:5.1f}%  "
          f"recall st/si/ly {rec[0]:4.1f}/{rec[1]:4.1f}/{rec[2]:4.1f}  precision {pre[0]:4.1f}/{pre[1]:4.1f}/{pre[2]:4.1f}")
    return m


def explain_misses(misses, skipped, excluded, tol):
    """Attach the most likely reason to each ground-truth cell that got no prediction."""
    out = []
    for g in misses:
        row = {"time": g["time"].strftime("%Y-%m-%d %H:%M"), "pig_id": g["pig_id"], "ground_truth": g["label"],
               "reason": "no_frame", "frame": "", "median_depth": "", "pig_present_prob": "", "image_path": ""}
        s = nearest(skipped.get(g["pig_id"], []), g["time"], tol) if skipped is not None else None
        if s is not None:
            row.update(reason=s["reason"], frame=s["pig_timestamp"], median_depth=s.get("median_depth", ""),
                       pig_present_prob=s.get("pig_present_prob", ""), image_path=s.get("image_path", ""))
        elif excluded is not None:
            e = nearest(excluded.get(g["pig_id"], []), g["time"], tol)
            if e is not None:
                row.update(reason="id_policy", frame=e["pig_timestamp"], median_depth=e.get("median_depth", ""),
                           image_path=e.get("image_path", ""))
        out.append(row)
    return out


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
    ap.add_argument("--tolerance", type=float, default=3.0,
                    help="Max minutes between GT and frame (Lucas's rows are ~5.5 min apart)")
    ap.add_argument("--stall-offset", type=int, default=0,
                    help="Subtract from GT pig numbers if the sheet counts stalls from 1")
    ap.add_argument("--sheets", nargs="*", default=None)
    ap.add_argument("--inspect", action="store_true", help="Print the first rows of each sheet and exit")
    ap.add_argument("--out", default=os.path.join(OUTPUT_DIR, "ground_truth_eval.csv"))
    ap.add_argument("--match", choices=["pass", "capture"], default="pass",
                    help="pass (default): match Lucas's row time to the start of the robot pass; "
                         "capture: to each stall's own capture time")
    ap.add_argument("--skipped", default=None,
                    help="skipped_frames.csv from the same run (default: next to --pred)")
    ap.add_argument("--excluded", default=None,
                    help="predictions_excluded.csv from the same run (default: next to --pred)")
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
    time_key = "timestamp_folder" if args.match == "pass" else "pig_timestamp"
    preds = read_predictions(args.pred, time_key)
    span = [t for seq in preds.values() for t, _ in seq]
    if span:
        from datetime import timedelta as _td
        lo, hi = min(span) - _td(minutes=args.tolerance), max(span) + _td(minutes=args.tolerance)
        outside = sum(1 for g in gt if not lo <= g["time"] <= hi)
        gt = [g for g in gt if lo <= g["time"] <= hi]
        if outside:
            print(f"  {outside} ground-truth cells fall outside the predicted period "
                  f"({lo:%m-%d %H:%M} .. {hi:%m-%d %H:%M}) and are left out")
    has_smooth = any((r.get("smoothed_prediction_name") or "").strip()
                     for seq in preds.values() for _, r in seq)

    # Each GT row -> nearest pass within tolerance. Lucas logs ~2 rows per pass, so when two rows
    # land on the same pass only the closest is scored (one comparison per pass and stall).
    cand, misses = {}, []
    for g in gt:
        seq = preds.get(g["pig_id"], [])
        times = [t for t, _ in seq]
        import bisect
        i = bisect.bisect_left(times, g["time"])
        best = None
        for k in (i - 1, i):
            if 0 <= k < len(seq):
                d = abs((times[k] - g["time"]).total_seconds())
                if d <= args.tolerance * 60 and (best is None or d < best[0]):
                    best = (d, seq[k])
        if best is None:
            misses.append(g)
            continue
        key = (g["pig_id"], best[1][0])
        if key not in cand or best[0] < cand[key][0]:
            cand[key] = (best[0], g, best[1][1])

    rows, raw_pairs, sm_pairs = [], [], []
    per_pig = defaultdict(lambda: [0, 0])
    for (pig, pass_t), (_, g, p) in sorted(cand.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        raw = p["prediction_name"]
        sm = p.get("smoothed_prediction_name") or raw
        raw_pairs.append((g["label"], raw))
        sm_pairs.append((g["label"], sm))
        per_pig[pig][0] += int(raw == g["label"])
        per_pig[pig][1] += 1
        rows.append({"time": g["time"].strftime("%Y-%m-%d %H:%M"), "pass": pass_t.strftime("%Y-%m-%d %H:%M:%S"),
                     "pig_id": pig, "ground_truth": g["label"], "prediction": raw, "smoothed": sm,
                     "frame": p["pig_timestamp"], "confidence": p.get("confidence", ""),
                     "gt_sheet": g.get("sheet", ""), "gt_time_source": g.get("time_source", "")})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["time"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nGround-truth cells: {len(gt)} | scored (one per pass and stall, matched by "
          f"{'pass start' if args.match == 'pass' else 'capture'} time within {args.tolerance:g} min): {len(rows)}")
    class_line("raw", raw_pairs)
    if has_smooth:
        class_line("smoothed", sm_pairs)
    night = lambda r: int(r["pass"][11:13]) >= 19 or int(r["pass"][11:13]) < 8
    class_line("night 19:00-08:00", [(r["ground_truth"], r["prediction"]) for r in rows if night(r)])
    class_line("day 08:00-19:00", [(r["ground_truth"], r["prediction"]) for r in rows if not night(r)])
    print(f"  confusion (rows = true {NAMES}): {metrics(raw_pairs)['confusion']}")
    print("  per-pig accuracy: " + ", ".join(f"{p}:{c / n * 100:.0f}%" for p, (c, n) in sorted(per_pig.items())))
    if per_pig and min(c / n for c, n in per_pig.values()) < 0.6:
        print("  Note: a pig far below the others usually means a camera-ID / stall shift on those days, "
              "not a model error — check data_quality.csv for those folders.")

    # nightly posture transitions per sow (the estrus signal), on the scored pass grid
    seqs = defaultdict(list)
    for r in rows:
        if night(r):
            t = datetime.strptime(r["pass"], "%Y-%m-%d %H:%M:%S")
            seqs[(r["pig_id"], (t + timedelta(hours=5)).date())].append((t, r))
    tt, tp, ts = [], [], []
    for seq in seqs.values():
        seq.sort(key=lambda x: x[0])
        if len(seq) < 40:
            continue
        a = b = c = 0
        for (t1, r1), (t2, r2) in zip(seq, seq[1:]):
            if (t2 - t1).total_seconds() > 15 * 60:
                continue
            a += r1["ground_truth"] != r2["ground_truth"]
            b += r1["prediction"] != r2["prediction"]
            c += r1["smoothed"] != r2["smoothed"]
        tt.append(a); tp.append(b); ts.append(c)
    if tt:
        mae = lambda x: sum(abs(u - v) for u, v in zip(tt, x)) / len(tt)
        print(f"\nNightly posture transitions per sow ({len(tt)} sow-nights with >= 40 scored passes):")
        print(f"  truth {sum(tt)/len(tt):.1f}/night | predicted {sum(tp)/len(tp):.1f} "
              f"(r = {pearson(tt, tp):.2f}, MAE {mae(tp):.1f})"
              + (f" | smoothed {sum(ts)/len(ts):.1f} (r = {pearson(tt, ts):.2f}, MAE {mae(ts):.1f})" if has_smooth else ""))
    print(f"\nRow-level comparison: {args.out}")

    # ── presence check: every GT cell has a sow, so an unmatched cell is a miss ──
    pred_dir = os.path.dirname(os.path.abspath(args.pred))
    skipped = read_frames(args.skipped or os.path.join(pred_dir, "skipped_frames.csv"), time_key=time_key)
    excluded = read_frames(args.excluded or os.path.join(pred_dir, "predictions_excluded.csv"), include_only=False,
                           time_key=time_key)
    # a GT row with no pass within tolerance just falls between two passes: not a miss
    passes = sorted({t for d in (preds, skipped or {}, excluded or {}) for seq in d.values() for t, _ in seq})
    import bisect

    def has_pass(t):
        i = bisect.bisect_left(passes, t)
        return any(0 <= k < len(passes) and abs((passes[k] - t).total_seconds()) <= args.tolerance * 60
                   for k in (i - 1, i))
    between = [g for g in misses if not has_pass(g["time"])]
    misses = [g for g in misses if has_pass(g["time"])]
    if skipped is None:
        print("\n  (no skipped_frames.csv next to --pred: re-run run_inference.py v1.1 to explain the misses)")
    explained = explain_misses(misses, skipped, excluded, args.tolerance)
    miss_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "ground_truth_misses.csv")
    with open(miss_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "pig_id", "ground_truth", "reason", "frame", "median_depth",
                                          "pig_present_prob", "image_path"])
        w.writeheader()
        w.writerows(explained)
    matched_by = defaultdict(int)
    for r in rows:
        matched_by[r["ground_truth"]] += 1
    miss_by = defaultdict(lambda: defaultdict(int))
    for r in explained:
        miss_by[r["ground_truth"]][r["reason"]] += 1
    print(f"\nPresence check (Lucas saw a sow; {len(between)} rows fall between passes and are not counted):")
    print(f"  {'posture':<9} {'cells':>6} {'predicted':>9}  " + "  ".join(f"{m:>9}" for m in MISS_REASONS)
          + "   kept / captured")
    for lab in NAMES:
        n_m = matched_by[lab]
        n_tot = n_m + sum(miss_by[lab].values())
        if not n_tot:
            continue
        captured = n_m + sum(miss_by[lab][m] for m in ("depth", "detector", "corrupt"))
        rate = f"{n_m / captured * 100:.1f}%" if captured else "-"
        print(f"  {lab:<9} {n_tot:>6} {n_m:>9}  " + "  ".join(f"{miss_by[lab][m]:>9}" for m in MISS_REASONS)
              + f"   {rate}")
    depth_lying = [float(r["median_depth"]) for r in explained
                   if r["reason"] == "depth" and r["ground_truth"] == "lying" and r["median_depth"] != ""]
    if depth_lying:
        depth_lying.sort()
        print(f"  lying sows called empty by the depth gate: median depth {depth_lying[len(depth_lying)//2]:.0f} mm "
              f"(range {depth_lying[0]:.0f}-{depth_lying[-1]:.0f})")
    print(f"  Misses with the frame each one maps to: {miss_path}")
    print("  Look at them with: python posture/review_presence.py --misses " + miss_path)


if __name__ == "__main__":
    main()
