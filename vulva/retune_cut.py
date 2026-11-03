#!/usr/bin/env python3
r"""Re-set a model's "vulva visible" confidence cut from its saved cross-validation predictions,
without retraining.

train_seg.py sets the cut so out-of-fold precision reaches 0.9 on all training frames. When part
of the training labels were picked on purpose because the model struggled on them (pick_hard_frames.py:
mostly tails and unsure vulvas), those frames are far harder than normal standing frames, and a
cut set on them is too strict for normal data. This re-sets the cut on the out-of-fold predictions
of the normally sampled frames only (--exclude-labels: the hand-picked set), and prints the
trade-off for a few precision targets so the choice is visible. Blind test sets are not used.

    python vulva/retune_cut.py outputs\vulva_seg_r9\cv_predictions.csv --model models\vulva_seg_r9.pth ^
        --exclude-labels C:\Users\aakdbx\vulva_r9_set\labels.csv --precision 0.9 --write models\vulva_seg_r9c.pth

--write saves a copy of the checkpoint with the new cut (the old one is kept as
presence_threshold_train_all); the original file is not changed.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def key(r):
    return str(int(_f(r.get("pig_id"), -1))), str(r.get("pig_timestamp", ""))


def read_cv(path, exclude=frozenset()):
    """cv_predictions.csv -> [(truth_present, drawn, conf)] for frames not in exclude."""
    out = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if key(r) in exclude or r.get("label") not in ("present", "absent"):
                continue
            out.append((r["label"] == "present", int(_f(r.get("pred_area_px"))) > 0, _f(r.get("blob_conf"))))
    return out


def at_cut(frames, cut):
    tp = sum(1 for t, d, c in frames if t and d and c >= cut)
    fp = sum(1 for t, d, c in frames if not t and d and c >= cut)
    pos = sum(1 for t, _, _ in frames if t)
    neg = len(frames) - pos
    return {"precision": tp / (tp + fp) if tp + fp else float("nan"), "recall": tp / pos if pos else float("nan"),
            "specificity": (neg - fp) / neg if neg else float("nan"), "tp": tp, "fp": fp, "present": pos}


def cut_for(frames, target):
    """Lowest cut whose precision reaches target (same rule as train_seg.pick_presence_cut)."""
    for c in sorted({round(c, 4) for t, d, c in frames if d}):
        s = at_cut(frames, c)
        if s["tp"] + s["fp"] and s["precision"] >= target:
            return c
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cv_csv", help="cv_predictions.csv from train_seg.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--exclude-labels", nargs="*", default=[], help="labels CSVs of hand-picked (hard) frames")
    ap.add_argument("--precision", type=float, default=0.9)
    ap.add_argument("--write", default="", help="save a checkpoint copy with the new cut")
    args = ap.parse_args()

    exclude = set()
    for p in args.exclude_labels:
        with open(p, newline="", encoding="utf-8-sig") as f:
            exclude |= {key(r) for r in csv.DictReader(f)}
    all_frames = read_cv(args.cv_csv)
    frames = read_cv(args.cv_csv, exclude)
    import torch
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    old = float(ckpt.get("presence_threshold", 0.0))
    print(f"{len(all_frames)} out-of-fold frames; {len(frames)} normally sampled ones after leaving out "
          f"{len(all_frames) - len(frames)} hand-picked")
    print(f"Current cut {old} (set on all training frames):")
    for name, fr in (("  all frames      ", all_frames), ("  normal frames   ", frames)):
        s = at_cut(fr, old)
        print(f"{name} precision {s['precision']:.3f}, recall {s['recall']:.3f}, specificity {s['specificity']:.3f}")
    print("Cut set on the normal frames, for a few precision targets (out-of-fold on those frames):")
    for target in sorted({0.8, 0.85, 0.9, 0.95, args.precision}):
        c = cut_for(frames, target)
        if c is None:
            print(f"  target {target}: not reachable")
            continue
        s = at_cut(frames, c)
        print(f"  target {target}: cut {c}  precision {s['precision']:.3f}, recall {s['recall']:.3f}, "
              f"specificity {s['specificity']:.3f} ({s['tp']} of {s['present']} vulvas kept, {s['fp']} false)")
    new = cut_for(frames, args.precision)
    if args.write:
        if new is None:
            sys.exit(f"No cut reaches precision {args.precision} on the normal frames; nothing written")
        ckpt["presence_threshold_train_all"] = old
        ckpt["presence_threshold"] = new
        ckpt["presence_cut_note"] = (f"re-set on {len(frames)} normally sampled out-of-fold frames "
                                     f"(precision {args.precision}); hand-picked frames left out")
        torch.save(ckpt, args.write)
        print(f"Saved {args.write} with cut {new} (was {old})")


if __name__ == "__main__":
    main()
