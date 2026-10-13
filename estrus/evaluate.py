#!/usr/bin/env python3
"""How well does each input pick out heat days? Leave-one-sow-out, simplest first.

Reads sow_day_features.csv (estrus/sow_days.py) and reports:

1. Per feature: AUC for heat vs non-heat sow-days, and the median on each side.
   0.5 = no information, 1.0 = perfect separation.
2. Logistic regression trained on all sows but one and tested on the one left
   out, repeated for every sow, for several input sets:
     days_from_move              (the reference everything must beat)
     posture                     (night transitions + night standing share, change from own baseline)
     days_from_move + posture
     ... + vulva                 (when vulva columns are filled)
   For each: AUC, and at a 0.5 cut sensitivity, specificity and precision per
   sow-day, plus how many days early or late the first predicted heat day is
   compared with the first recorded heat day.

Only sow-days that have every input in the set are scored, and the count is
printed, so sets are compared on what they actually saw.

Usage:
    python estrus/evaluate.py outputs/estrus/sow_day_features.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics

import numpy as np

POSTURE = ["night_transitions_delta", "night_standing_frac_delta"]
FEATURE_SETS = {
    "days_from_move": ["dfw"],
    "posture": POSTURE,
    "days_from_move + posture": ["dfw"] + POSTURE,
    "vulva": ["vulva_index_delta"],
    "days_from_move + vulva": ["dfw", "vulva_index_delta"],
    "days_from_move + posture + vulva": ["dfw"] + POSTURE + ["vulva_index_delta"],
}
SINGLE = ["dfw", "night_transitions", "night_transitions_delta", "night_transitions_per_h",
          "night_standing_frac", "night_standing_frac_delta", "night_lying_frac_delta",
          "day_standing_frac", "day_standing_frac_delta", "vulva_index", "vulva_index_delta"]


def _f(v):
    try:
        x = float(v)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def auc(pos, neg):
    """Mann-Whitney AUC; ties count half."""
    if not pos or not neg:
        return None
    s = 0.0
    for p in pos:
        for n in neg:
            s += 1.0 if p > n else 0.5 if p == n else 0.0
    return s / (len(pos) * len(neg))


def fit_logistic(X, y, l2=1.0, iters=50):
    """Newton's method with a small ridge penalty; X already standardised."""
    X1 = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(X1.shape[1])
    reg = np.eye(len(w)) * l2
    reg[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ w))
        g = X1.T @ (p - y) + reg @ w
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 + reg
        step = np.linalg.solve(H + 1e-9 * np.eye(len(w)), g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict(w, X):
    X1 = np.hstack([np.ones((len(X), 1)), X])
    return 1 / (1 + np.exp(-X1 @ w))


def loso(rows, cols, cut=0.5):
    data = [r for r in rows if all(_f(r.get(c)) is not None for c in cols)]
    pigs = sorted({r["pig_id"] for r in data})
    if len(pigs) < 3 or len({r["heat"] for r in data}) < 2:
        return None
    probs = {}
    for p in pigs:
        tr = [r for r in data if r["pig_id"] != p]
        te = [r for r in data if r["pig_id"] == p]
        Xtr = np.array([[_f(r[c]) for c in cols] for r in tr])
        ytr = np.array([r["heat"] for r in tr], float)
        if len(set(ytr)) < 2:
            continue
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        w = fit_logistic((Xtr - mu) / sd, ytr)
        Xte = np.array([[_f(r[c]) for c in cols] for r in te])
        for r, pr in zip(te, predict(w, (Xte - mu) / sd)):
            probs[(r["pig_id"], r["date"])] = float(pr)
    scored = [r for r in data if (r["pig_id"], r["date"]) in probs]
    y = [r["heat"] for r in scored]
    pr = [probs[(r["pig_id"], r["date"])] for r in scored]
    tp = sum(1 for a, b in zip(y, pr) if a and b >= cut)
    fp = sum(1 for a, b in zip(y, pr) if not a and b >= cut)
    fn = sum(1 for a, b in zip(y, pr) if a and b < cut)
    tn = sum(1 for a, b in zip(y, pr) if not a and b < cut)
    # onset timing per sow: first predicted heat day vs first recorded heat day
    errs = []
    for p in pigs:
        rs = sorted((r for r in scored if r["pig_id"] == p), key=lambda r: r["dfw"])
        true_on = [r["dfw"] for r in rs if r["heat"]]
        pred_on = [r["dfw"] for r in rs if probs[(p, r["date"])] >= cut]
        if true_on and pred_on:
            errs.append(pred_on[0] - true_on[0])
    return {
        "sow_days": len(scored), "sows": len(pigs), "heat_days": sum(y),
        "auc": round(auc([b for a, b in zip(y, pr) if a], [b for a, b in zip(y, pr) if not a]) or 0, 3),
        "sensitivity": round(tp / (tp + fn), 3) if tp + fn else None,
        "specificity": round(tn / (tn + fp), 3) if tn + fp else None,
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "onset_error_days_mean_abs": round(statistics.mean(abs(e) for e in errs), 2) if errs else None,
        "onset_error_days": errs,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", help="sow_day_features.csv")
    ap.add_argument("--out", default=os.path.join("outputs", "estrus", "evaluation.json"))
    args = ap.parse_args()

    with open(args.features, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["pig_id"] = int(float(r["pig_id"]))
        r["heat"] = int(float(r["heat"]))
        r["dfw"] = int(float(r["dfw"]))
    report = {"features": {}, "models": {}}

    print(f"{len(rows)} sow-days, {sum(r['heat'] for r in rows)} in heat, "
          f"{len({r['pig_id'] for r in rows})} sows\n")
    print(f"{'feature':32s} {'AUC':>5s} {'n':>4s}  median heat / other")
    for c in SINGLE:
        pos = [_f(r.get(c)) for r in rows if r["heat"] and _f(r.get(c)) is not None]
        neg = [_f(r.get(c)) for r in rows if not r["heat"] and _f(r.get(c)) is not None]
        a = auc(pos, neg)
        if a is None:
            continue
        report["features"][c] = {"auc": round(a, 3), "n": len(pos) + len(neg),
                                 "median_heat": statistics.median(pos), "median_other": statistics.median(neg)}
        print(f"{c:32s} {a:5.2f} {len(pos) + len(neg):4d}  {statistics.median(pos):.3g} / {statistics.median(neg):.3g}")

    print("\nLeave-one-sow-out logistic regression (cut 0.5):")
    print(f"{'inputs':34s} {'AUC':>5s} {'sens':>5s} {'spec':>5s} {'prec':>5s} {'n':>4s}  onset error (days)")
    for name, cols in FEATURE_SETS.items():
        res = loso(rows, cols)
        if res is None:
            continue
        report["models"][name] = res
        fmt = lambda v: f"{v:5.2f}" if v is not None else "  n/a"  # noqa: E731
        print(f"{name:34s} {fmt(res['auc'])} {fmt(res['sensitivity'])} {fmt(res['specificity'])} "
              f"{fmt(res['precision'])} {res['sow_days']:4d}  {res['onset_error_days_mean_abs']}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
