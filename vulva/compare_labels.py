#!/usr/bin/env python3
r"""How much do different people's labels agree? Every pair of labels CSVs is compared on the
frames both labelled (normally an agreement set from make_agreement_set.py + import_cvat.py):

  - visible or not: how often they agree, and Cohen's kappa (agreement beyond chance)
  - outline: IoU on frames both call visible (mean, median, share below 0.5) and the size ratio
  - "human mAP": the second person's outlines scored against the first person's with the same
    mAP50 / mAP50-95 as the models (eval_seg.py). If two people agree at 0.6, no model trained on
    one person's labels should be expected to reach 0.8 against the other's.
  - with --mm: width and length in mm from the depth (measure3d.py) for both outlines

    python vulva\compare_labels.py "$FEB\vulva_test_feb12\labels.csv" C:\Users\aakdbx\agreement_set\labels_lucas.csv C:\Users\aakdbx\agreement_set\labels_pushpa.csv --names akbar,lucas,pushpa --manifest C:\Users\aakdbx\agreement_set\manifest.csv --frames-root $RAW --mm

Writes compare_report.txt next to the manifest (or the second CSV).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_seg import IOU_THRESHOLDS, read_rows  # noqa: E402
from seg_model import read_mask  # noqa: E402


def key(r):
    try:
        return str(int(float(r.get("pig_id")))), str(r.get("pig_timestamp", ""))
    except (TypeError, ValueError):
        return str(r.get("pig_id")), str(r.get("pig_timestamp", ""))


def load(path, frames_roots=None, keep=None):
    """labels CSV -> {frame key: {"present", "tail", "mask", "row"}}"""
    out = {}
    for r in read_rows(path, frames_roots):
        k = key(r)
        if keep is not None and k not in keep:
            continue
        present = r["label"] == "present"
        m = read_mask(r["mask_path"]) if present else None
        if present and (m is None or not m.any()):
            continue
        out[k] = {"present": present, "tail": "tail" in (r.get("note") or ""), "mask": m, "row": r}
    return out


def kappa(a, b):
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def human_map(ious, n_ref, n_other):
    """AP of person B's outlines against person A's: all of B's outlines carry the same score, so the
    precision-recall curve is one point (precision P, recall R) and COCO's 101-point AP is P x the
    share of recall points up to R. Returns (mAP50, mAP50-95)."""
    aps = []
    for t in IOU_THRESHOLDS:
        tp = sum(1 for i in ious if i >= t)
        p = tp / n_other if n_other else 0.0
        r = tp / n_ref if n_ref else 0.0
        aps.append(p * (int(np.floor(r * 100 + 1e-9)) + 1) / 101 if tp else 0.0)
    return aps[0], float(np.mean(aps))


def compare(a, b, mm=False):
    common = sorted(set(a) & set(b))
    va = [a[k]["present"] for k in common]
    vb = [b[k]["present"] for k in common]
    both = [k for k in common if a[k]["present"] and b[k]["present"]]
    ious, ratios, dw, dl = [], [], [], []
    for k in both:
        x, y = a[k]["mask"] > 0, b[k]["mask"] > 0
        ious.append(float((x & y).sum() / max((x | y).sum(), 1)))
        ratios.append(float(y.sum() / max(x.sum(), 1)))
    if mm:
        from measure3d import load_depth_raw, measure
        from seg_model import default_depth_path
        for k in both:
            ir = a[k]["row"]["ir_path"]
            dp = a[k]["row"].get("depth_path") or default_depth_path(ir)
            if not os.path.exists(dp):
                continue
            depth = load_depth_raw(dp)
            ma, mb = measure(depth, a[k]["mask"]), measure(depth, b[k]["mask"])
            if ma["measure_status"] == "ok" and mb["measure_status"] == "ok":
                dw.append(mb["width_mm"] - ma["width_mm"])
                dl.append(mb["length_mm"] - ma["length_mm"])
    n_ref = sum(1 for k in common if a[k]["present"])
    n_oth = sum(1 for k in common if b[k]["present"])
    m50, m5095 = human_map(ious, n_ref, n_oth)
    return {"frames": len(common), "both_visible": len(both),
            "only_first": sum(1 for x, y in zip(va, vb) if x and not y),
            "only_second": sum(1 for x, y in zip(va, vb) if y and not x),
            "both_not": sum(1 for x, y in zip(va, vb) if not x and not y),
            "agree": sum(x == y for x, y in zip(va, vb)) / len(common) if common else float("nan"),
            "kappa": kappa(va, vb), "ious": ious, "ratios": ratios, "dw": dw, "dl": dl,
            "map50": m50, "map5095": m5095,
            "tail_agree": sum(1 for k in common if a[k]["tail"] == b[k]["tail"]) / len(common) if common else 0}


def report(name_a, name_b, s):
    lines = [f"{name_b} against {name_a}: {s['frames']} frames labelled by both"]
    lines.append(f"  visible or not: agree on {s['agree']:.0%} (kappa {s['kappa']:.2f}); both visible "
                 f"{s['both_visible']}, only {name_a} {s['only_first']}, only {name_b} {s['only_second']}, "
                 f"both not {s['both_not']}; tail marks agree on {s['tail_agree']:.0%}")
    if s["ious"]:
        iou = np.array(s["ious"])
        lines.append(f"  outline overlap (IoU) on frames both call visible: mean {iou.mean():.3f}, "
                     f"median {np.median(iou):.3f}, below 0.5 on {int((iou < 0.5).sum())}; {name_b}'s outline is "
                     f"{np.median(s['ratios']):.2f} x the area of {name_a}'s (median)")
        lines.append(f"  human mAP50 {s['map50']:.3f}, mAP50-95 {s['map5095']:.3f}  (same metric as the models)")
    if s["dw"]:
        lines.append(f"  size, {name_b} minus {name_a} ({len(s['dw'])} frames measured): width "
                     f"{np.mean(s['dw']):+.1f} mm (mean abs {np.mean(np.abs(s['dw'])):.1f}), length "
                     f"{np.mean(s['dl']):+.1f} mm (mean abs {np.mean(np.abs(s['dl'])):.1f})")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="+", help="two or more labels CSVs; the first is the reference")
    ap.add_argument("--names", default="", help="comma names, in the same order (default: file names)")
    ap.add_argument("--manifest", default="", help="manifest.csv of an agreement set: compare only its frames")
    ap.add_argument("--frames-root", default="", help="';'-separated raw folders (images and depth not next to labels)")
    ap.add_argument("--mm", action="store_true", help="also compare width and length in mm (needs the depth files)")
    args = ap.parse_args()
    roots = [r for r in args.frames_root.split(";") if r.strip()]
    keep = None
    if args.manifest:
        with open(args.manifest, newline="", encoding="utf-8-sig") as f:
            keep = {key(r) for r in csv.DictReader(f)}
    names = [n.strip() for n in args.names.split(",")] if args.names else \
        [os.path.splitext(os.path.basename(p))[0] for p in args.labels]
    sets = [load(p, roots, keep) for p in args.labels]
    lines = []
    for (na, a), (nb, b) in itertools.combinations(zip(names, sets), 2):
        lines += report(na, nb, compare(a, b, args.mm))
    print("\n".join(lines))
    out_dir = os.path.dirname(os.path.abspath(args.manifest or args.labels[1]))
    with open(os.path.join(out_dir, "compare_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
