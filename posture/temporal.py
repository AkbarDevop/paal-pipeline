"""Temporal post-processing for per-pig posture sequences.

A sow is photographed roughly every 10 minutes. A single misclassified frame
inside a long lying bout creates two fake transitions (lying->X->lying), so
frame-level errors inflate the nighttime transition counts used for estrus
detection. smooth_sequence() removes isolated one-frame flips with a centered
majority vote, but only across frames that are close in time.
"""

from collections import Counter
from datetime import datetime

TS_FORMAT = "%Y%m%d-%H-%M-%S"


def parse_ts(ts):
    try:
        return datetime.strptime(ts, TS_FORMAT)
    except (TypeError, ValueError):
        return None


def _segments(times, max_gap_minutes):
    """Split indices into runs where consecutive frames are <= max_gap apart."""
    if not times:
        return []
    segs, start = [], 0
    for i in range(1, len(times)):
        gap = (times[i] - times[i - 1]).total_seconds() / 60.0
        if gap > max_gap_minutes:
            segs.append((start, i))
            start = i
    segs.append((start, len(times)))
    return segs


def smooth_sequence(labels, times, window=3, max_gap_minutes=30, probs=None):
    """Centered majority vote over `window` frames inside gap-free segments.

    labels: list of class names (time-ordered)
    times:  list of datetimes, same order
    probs:  optional list of per-frame probability dicts; ties are broken by
            summed probability, otherwise the original label is kept.
    window=1 (or less) returns the labels unchanged.
    """
    n = len(labels)
    if window <= 1 or n == 0:
        return list(labels)
    half = window // 2
    out = list(labels)
    for s, e in _segments(times, max_gap_minutes):
        for i in range(s, e):
            lo, hi = max(s, i - half), min(e, i + half + 1)
            votes = Counter(labels[lo:hi])
            top = votes.most_common()
            best_n = top[0][1]
            tied = [lab for lab, c in top if c == best_n]
            if len(tied) == 1:
                out[i] = tied[0]
            elif labels[i] in tied:
                out[i] = labels[i]
            elif probs is not None:
                out[i] = max(tied, key=lambda lab: sum(p.get(lab, 0.0) for p in probs[lo:hi]))
    return out


def smooth_results(results, window=3, max_gap_minutes=30, key="prediction_name",
                   out_key="smoothed_prediction_name", class_names=("standing", "sitting", "lying")):
    """Add out_key to each included result row, smoothing per pig over time."""
    by_pig = {}
    for r in results:
        if not r.get("include", True):
            r[out_key] = r[key]
            continue
        by_pig.setdefault(r["pig_id"], []).append(r)
    for rows in by_pig.values():
        rows.sort(key=lambda r: r["pig_timestamp"])
        times = [parse_ts(r["pig_timestamp"]) for r in rows]
        ok = [t is not None for t in times]
        idx = [i for i, good in enumerate(ok) if good]
        labels = [rows[i][key] for i in idx]
        probs = [{c: float(rows[i].get(f"prob_{c}", 0.0)) for c in class_names} for i in idx]
        smoothed = smooth_sequence(labels, [times[i] for i in idx], window, max_gap_minutes, probs)
        for i, lab in zip(idx, smoothed):
            rows[i][out_key] = lab
        for i, good in enumerate(ok):
            if not good:
                rows[i][out_key] = rows[i][key]
    return results
