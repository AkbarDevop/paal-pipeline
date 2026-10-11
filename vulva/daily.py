"""Per-frame vulva measurements -> one robust value per sow per day.

The estrus change is small (calipers: about +9% width, +10% length, +20% for
the W x L index), so single frames are too noisy. A day's value is the median
of that sow's good frames, compared with her own baseline days.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

METRICS = ["width_mm", "length_mm", "peak_height_mm", "volume_mm3", "vulva_index"]

# frames worse than this are not used for the daily value
QUALITY = {"max_plane_rmse_mm": 6.0, "max_view_angle_deg": 70.0, "min_depth_coverage": 0.5}


def frame_date(row):
    ts = str(row.get("timestamp_folder") or row.get("pig_timestamp") or "")
    return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 and ts[:8].isdigit() else ""


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def is_good(row, quality=QUALITY):
    status = str(row.get("measure_status") or row.get("status") or "")
    if status not in ("ok", "measured"):
        return False
    rmse, ang, cov = _f(row.get("plane_rmse_mm")), _f(row.get("view_angle_deg")), _f(row.get("depth_coverage"))
    if rmse is not None and rmse > quality["max_plane_rmse_mm"]:
        return False
    if ang is not None and ang > quality["max_view_angle_deg"]:
        return False
    if cov is not None and cov < quality["min_depth_coverage"]:
        return False
    w, l = _f(row.get("width_mm")), _f(row.get("length_mm"))
    return w is not None and l is not None and w > 0 and l > 0


def mad(xs):
    m = statistics.median(xs)
    return statistics.median(abs(x - m) for x in xs)


def aggregate(rows, quality=QUALITY, min_frames=1):
    """rows: per-frame dicts with pig_id, timestamp_folder and METRICS.
    Returns one dict per (pig_id, date) with n_frames and median / MAD per metric."""
    groups = defaultdict(list)
    for r in rows:
        if not is_good(r, quality):
            continue
        r = dict(r)
        if _f(r.get("vulva_index")) is None:
            r["vulva_index"] = _f(r["width_mm"]) * _f(r["length_mm"]) / 200.0
        groups[(int(float(r["pig_id"])), frame_date(r))].append(r)
    out = []
    for (pid, day), rs in sorted(groups.items()):
        if len(rs) < min_frames:
            continue
        rec = {"pig_id": pid, "date": day, "n_frames": len(rs)}
        for k in METRICS:
            vals = [v for v in (_f(r.get(k)) for r in rs) if v is not None]
            if vals:
                rec[k] = round(statistics.median(vals), 3)
                rec[k + "_mad"] = round(mad(vals), 3)
        out.append(rec)
    return out


def add_baseline_change(daily, baseline_days=2):
    """% change of each metric from the median of the sow's first baseline_days days."""
    by = defaultdict(list)
    for r in daily:
        by[r["pig_id"]].append(r)
    for rs in by.values():
        rs.sort(key=lambda r: r["date"])
        base = rs[:baseline_days]
        for k in METRICS:
            vals = [r[k] for r in base if k in r]
            if not vals:
                continue
            b = statistics.median(vals)
            for r in rs:
                if k in r and b:
                    r[k + "_change_pct"] = round(100 * (r[k] - b) / b, 2)
    return daily


def pearson(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def compare_truth(daily, truth):
    """Join camera daily values with Lucas's caliper rows (pig_id, date).
    Returns (joined rows, summary dict)."""
    t = {(int(r["pig_id"]), str(r["date"])): r for r in truth}
    joined = []
    for r in daily:
        g = t.get((r["pig_id"], r["date"]))
        if not g:
            continue
        j = dict(r)
        j["caliper_width_mm"] = _f(g.get("vulva_width_mm"))
        j["caliper_length_mm"] = _f(g.get("vulva_length_mm"))
        j["caliper_index"] = _f(g.get("vulva_index"))
        j["heat"] = g.get("heat")
        joined.append(j)
    summary = {"sow_days": len(joined)}
    for cam, cal in (("width_mm", "caliper_width_mm"), ("length_mm", "caliper_length_mm"),
                     ("vulva_index", "caliper_index")):
        pairs = [(j[cam], j[cal]) for j in joined if j.get(cam) is not None and j.get(cal) is not None]
        if pairs:
            summary[cam + "_mae"] = round(statistics.mean(abs(a - b) for a, b in pairs), 2)
            summary[cam + "_bias"] = round(statistics.mean(a - b for a, b in pairs), 2)
            r = pearson([a for a, _ in pairs], [b for _, b in pairs])
            summary[cam + "_r"] = round(r, 3) if r is not None else None
    heat_vals = [j.get("vulva_index_change_pct") for j in joined
                 if str(j.get("heat")) in ("1", "True") and j.get("vulva_index_change_pct") is not None]
    other = [j.get("vulva_index_change_pct") for j in joined
             if str(j.get("heat")) in ("0", "False") and j.get("vulva_index_change_pct") is not None]
    if heat_vals and other:
        summary["index_change_heat_days"] = round(statistics.median(heat_vals), 2)
        summary["index_change_other_days"] = round(statistics.median(other), 2)
    return joined, summary
