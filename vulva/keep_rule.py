#!/usr/bin/env python3
r"""Keep or drop a predicted vulva: the model's confidence plus a check that the outline sits on a bump.

The confidence cut alone (train_seg.py, tuned for 0.9 precision on February sows) drops many real
vulvas on new sows. A real vulva sticks out of the skin, so a blob the model is only fairly sure of
is kept when the depth under it rises above the surrounding skin (peak_height_mm from measure3d.py):

    keep  if  conf >= conf_high
          or  conf >= conf_low  and  the blob was measured  and  peak_height_mm >= min_peak_mm

Without a tuned rule this is the old cut (conf_low = conf_high = the checkpoint's presence_threshold).

Tune it on blind tests (eval_seg.py eval_frames.csv, one per test). The rule with the highest recall
whose precision reaches --precision on every test is picked. A kept frame counts as right only when
it is labelled visible and the model's outline overlaps the hand outline (IoU >= --min-iou). The
report also tunes on each test alone and scores the other, as a check that the rule carries over.

    python vulva/keep_rule.py outputs/eval_r8_feb/eval_frames.csv outputs/eval_r8_jan/eval_frames.csv ^
        --model models/vulva_seg_r8_ir.pth --write models/vulva_seg_r8k.pth

--write saves a copy of the checkpoint with the rule in it ("keep_rule"); run_vulva.py and
eval_seg.py use it from there. The original checkpoint is not changed.

To re-apply a rule to a run made with run_vulva.py (every blob is measured, kept or not):
    python vulva/keep_rule.py --apply outputs/vulva_r8_long/vulva_frames.csv --model models/vulva_seg_r8k.pth
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def rule_from_checkpoint(ckpt):
    """The checkpoint's keep rule, or the plain confidence cut for older checkpoints."""
    rule = ckpt.get("keep_rule")
    if rule:
        return dict(rule)
    cut = float(ckpt.get("presence_threshold", 0.0))
    return {"conf_high": cut, "conf_low": cut, "min_peak_mm": 0.0}


def keep(conf, peak_mm, measured, rule):
    """True when a blob with this confidence and bump height is kept as a vulva."""
    if conf is None or conf <= 0:
        return False
    if conf >= rule["conf_high"]:
        return True
    return bool(conf >= rule["conf_low"] and measured and peak_mm is not None and peak_mm >= rule["min_peak_mm"])


def frames_from_eval(path):
    """eval_frames.csv -> list of dicts with truth, conf, peak, measured, iou."""
    out = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if "model_peak_height_mm" not in r:
                sys.exit(f"{path} has no model_peak_height_mm column: rerun eval_seg.py with this version")
            out.append({"present": r["label"] == "present",
                        "blob": r.get("model_visible_before_cut") == "1",
                        "conf": _f(r.get("blob_conf"), 0.0),
                        "peak": _f(r.get("model_peak_height_mm")),
                        "measured": r.get("model_status") == "ok",
                        "iou": _f(r.get("iou")),
                        "tail": "tail" in (r.get("note") or "")})
    return out


def score(frames, rule, min_iou=0.2):
    tp = fp = fn = tn = wrong = 0
    for f in frames:
        k = f["blob"] and keep(f["conf"], f["peak"], f["measured"], rule)
        if f["present"]:
            if k and (f["iou"] is None or f["iou"] >= min_iou):
                tp += 1
            elif k:
                wrong += 1                    # kept, but drawn somewhere else
                fn += 1
            else:
                fn += 1
        elif k:
            fp += 1
        else:
            tn += 1
    kept_bad = fp + wrong
    prec = tp / (tp + kept_bad) if tp + kept_bad else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {"tp": tp, "fp": fp, "wrong_place": wrong, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "specificity": spec}


def grid():
    highs = [round(x, 3) for x in np.concatenate([np.arange(0.5, 0.95, 0.05), np.arange(0.95, 1.0, 0.005)])]
    lows = [round(x, 3) for x in np.arange(0.3, 1.0, 0.05)]
    peaks = [round(x, 1) for x in np.arange(0.0, 20.5, 0.5)]
    for hi in highs:
        yield {"conf_high": hi, "conf_low": hi, "min_peak_mm": 0.0}           # confidence only
        for lo in lows:
            if lo >= hi:
                break
            for pk in peaks:
                yield {"conf_high": hi, "conf_low": lo, "min_peak_mm": pk}


def tune(tests, precision=0.95, min_iou=0.2):
    """tests: list of frame lists. Highest pooled recall with precision >= target on every test;
    ties go to the higher worst-test precision, then to the stricter rule."""
    best, best_key = None, None
    for rule in grid():
        scores = [score(t, rule, min_iou) for t in tests]
        precs = [s["precision"] for s in scores]
        if any(not (p >= precision) for p in precs if p == p) or all(p != p for p in precs):
            continue
        tp = sum(s["tp"] for s in scores)
        pos = sum(s["tp"] + s["fn"] for s in scores)
        key = (tp / max(pos, 1), min(p for p in precs if p == p), rule["conf_low"], rule["min_peak_mm"])
        if best_key is None or key > best_key:
            best, best_key = rule, key
    return best


def fmt(s):
    return (f"precision {s['precision']:.3f}, recall {s['recall']:.3f}, specificity {s['specificity']:.3f} "
            f"(kept {s['tp']} right, {s['fp']} false alarms, {s['wrong_place']} in the wrong place; "
            f"missed {s['fn']})")


def report(names, tests, rule, old, precision, min_iou):
    lines = [f"Keep rule: conf >= {rule['conf_high']}, or conf >= {rule['conf_low']} and bump >= "
             f"{rule['min_peak_mm']} mm" if rule["conf_low"] < rule["conf_high"] else
             f"Keep rule: conf >= {rule['conf_high']} (no bump branch helps)",
             f"  (tuned for precision >= {precision} on every test; a kept vulva counts only with IoU >= {min_iou})"]
    for n, t in zip(names, tests):
        lines.append(f"{n}:")
        lines.append(f"  old cut ({old['conf_high']}):  {fmt(score(t, old, min_iou))}")
        lines.append(f"  keep rule:        {fmt(score(t, rule, min_iou))}")
        tails = [f for f in t if f["tail"]]
        if tails:
            k = sum(1 for f in tails if f["blob"] and keep(f["conf"], f["peak"], f["measured"], rule))
            lines.append(f"  tail frames kept: {k} of {len(tails)}")
    if len(tests) == 2:
        lines.append("Carry-over check (tune on one test, score the other):")
        for i, j in ((0, 1), (1, 0)):
            r = tune([tests[i]], precision, min_iou)
            if r:
                lines.append(f"  tuned on {names[i]} -> {r}; on {names[j]}: {fmt(score(tests[j], r, min_iou))}")
    # how well the two numbers separate real vulvas from the rest, among blobs the model drew
    for n, t in zip(names, tests):
        blobs = [f for f in t if f["blob"] and f["measured"] and f["peak"] is not None]
        real = [f["peak"] for f in blobs if f["present"] and (f["iou"] or 0) >= min_iou]
        other = [f["peak"] for f in blobs if not (f["present"] and (f["iou"] or 0) >= min_iou)]
        if real and other:
            lines.append(f"{n} bump height of drawn blobs: real vulvas median {np.median(real):.1f} mm "
                         f"(n {len(real)}), other blobs median {np.median(other):.1f} mm (n {len(other)})")
    return lines


def apply_to_run(frames_csv, rule):
    """Re-decide kept / dropped in a run_vulva.py vulva_frames.csv and redo the daily table."""
    from daily import METRICS, add_baseline_change, aggregate
    from run_vulva import write_csv
    with open(frames_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields, rows = list(reader.fieldnames), list(reader)
    for r in rows:
        conf, status = _f(r.get("vulva_conf"), 0.0), r.get("blob_status") or r.get("measure_status", "")
        if conf <= 0 or r.get("measure_status") in ("unreadable_ir", "not_visible"):
            continue
        k = keep(conf, _f(r.get("peak_height_mm")), status == "ok", rule)
        r["vulva_visible"] = int(k)
        r["measure_status"] = status if k else "low_confidence"
    base = os.path.splitext(frames_csv)[0]
    write_csv(rows, base + "_rekept.csv", fields)
    daily = add_baseline_change(aggregate(rows, min_frames=3), 2)
    daily_fields = ["pig_id", "date", "n_frames"] + [k + s for k in METRICS for s in ("", "_mad", "_change_pct")]
    write_csv(daily, os.path.join(os.path.dirname(frames_csv), "vulva_daily_rekept.csv"), daily_fields)
    kept = sum(1 for r in rows if str(r.get("vulva_visible")) == "1")
    return kept, len(rows), len(daily)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("evals", nargs="*", help="eval_frames.csv from eval_seg.py, one per blind test")
    ap.add_argument("--names", default="", help="comma names for the tests (default: their folders)")
    ap.add_argument("--model", required=True, help="checkpoint the evals were made with")
    ap.add_argument("--precision", type=float, default=0.95, help="minimum precision on every test")
    ap.add_argument("--min-iou", type=float, default=0.2)
    ap.add_argument("--write", default="", help="save a copy of the checkpoint with the tuned rule")
    ap.add_argument("--apply", default="", help="vulva_frames.csv of a run to re-decide with the checkpoint's rule")
    args = ap.parse_args()

    import torch
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    if args.apply:
        rule = rule_from_checkpoint(ckpt)
        kept, n, days = apply_to_run(args.apply, rule)
        print(f"Rule {rule}: kept {kept} of {n} frames; {days} sow-days with at least 3 good frames. "
              f"Wrote {os.path.splitext(args.apply)[0]}_rekept.csv and vulva_daily_rekept.csv")
        return
    if not args.evals:
        sys.exit("Give eval_frames.csv files to tune on, or --apply a run")
    names = [n.strip() for n in args.names.split(",")] if args.names else \
        [os.path.basename(os.path.dirname(os.path.abspath(p))) or p for p in args.evals]
    tests = [frames_from_eval(p) for p in args.evals]
    old = rule_from_checkpoint({k: v for k, v in ckpt.items() if k != "keep_rule"})
    rule = tune(tests, args.precision, args.min_iou)
    if rule is None:
        sys.exit(f"No rule reaches precision {args.precision} on every test; try a lower --precision")
    lines = report(names, tests, rule, old, args.precision, args.min_iou)
    print("\n".join(lines))
    out_dir = os.path.dirname(os.path.abspath(args.evals[0]))
    with open(os.path.join(out_dir, "keep_rule_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "keep_rule.json"), "w") as f:
        json.dump({**rule, "precision_target": args.precision, "min_iou": args.min_iou,
                   "tuned_on": args.evals, "model": args.model}, f, indent=2)
    if args.write:
        ckpt["keep_rule"] = {**rule, "tuned_on": names, "precision_target": args.precision}
        torch.save(ckpt, args.write)
        print(f"Saved {args.write} (same weights, with the keep rule)")


if __name__ == "__main__":
    main()
