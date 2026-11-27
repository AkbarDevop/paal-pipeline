#!/usr/bin/env python3
r"""Frames of a dataset, ready to re-check in label_ellipse.py --review, so older outlines can be brought onto
the current outline rule without touching the dataset itself.

    python vulva\make_review_set.py C:\Users\aakdbx\vulva_dataset_v2\labels.csv C:\Users\aakdbx\review_tests ^
        --splits test_newdays,test_newsows
    python vulva\make_review_set.py C:\Users\aakdbx\vulva_dataset_v2\labels.csv C:\Users\aakdbx\review_train_old ^
        --splits train --skip-sources diverse_febmar,diverse_jan
    python vulva\label_ellipse.py C:\Users\aakdbx\review_tests --review rule1126
    python vulva\build_dataset.py C:\Users\aakdbx\vulva_dataset_v3 --from-dataset C:\Users\aakdbx\vulva_dataset_v2\labels.csv ^
        --relabel C:\Users\aakdbx\review_tests\labels.csv --relabel C:\Users\aakdbx\review_train_old\labels.csv

Writes OUT/label_set.csv (the frames, with absolute paths into the dataset: IR, depth, colour) and OUT/labels.csv
(their current labels and ellipses, review column empty). Nothing is copied and the dataset is not changed;
label_ellipse.py --review writes redrawn masks into OUT/masks, and build_dataset.py --relabel takes the reviewed
rows into the next dataset version.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

SET_FIELDS = ["timestamp_folder", "pig_id", "pig_timestamp", "ir_path", "depth_path", "posture_confidence",
              "rgb_aligned_path", "rgb_path"]
LABEL_FIELDS = ["ir_path", "mask_path", "label", "pig_id", "timestamp_folder", "pig_timestamp",
                "cx", "cy", "half_length_px", "half_width_px", "angle_deg", "how", "note", "review"]


def make_review_set(labels_csv, out_dir, splits=None, sources=None, skip_sources=None):
    base = os.path.dirname(os.path.abspath(labels_csv))
    ab = lambda p: (p if os.path.isabs(p) else os.path.join(base, p)) if p else ""  # noqa: E731
    with open(labels_csv, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("label") in ("present", "absent")
                and (not splits or r.get("split") in splits)
                and (not sources or r.get("source") in sources)
                and (not skip_sources or r.get("source") not in skip_sources)]
    colour = {}
    ls = os.path.join(base, "label_set.csv")
    if os.path.exists(ls):
        with open(ls, newline="", encoding="utf-8-sig") as f:
            colour = {c["ir_path"]: c for c in csv.DictReader(f)}
    os.makedirs(out_dir, exist_ok=True)
    frames, labels = [], []
    for r in rows:
        ir = ab(r["ir_path"])
        c = colour.get(r["ir_path"], {})
        frames.append({"timestamp_folder": r.get("timestamp_folder", ""), "pig_id": r.get("pig_id", ""),
                       "pig_timestamp": r.get("pig_timestamp", ""), "ir_path": ir,
                       "depth_path": ab(r.get("depth_path", "")), "posture_confidence": "",
                       "rgb_aligned_path": ab(c.get("rgb_aligned_path", "")), "rgb_path": ab(c.get("rgb_path", ""))})
        labels.append({**{k: r.get(k, "") for k in LABEL_FIELDS}, "ir_path": ir, "mask_path": ab(r.get("mask_path", "")),
                       "review": ""})
    for name, fields, data in (("label_set.csv", SET_FIELDS, frames), ("labels.csv", LABEL_FIELDS, labels)):
        p = os.path.join(out_dir, name)
        if name == "labels.csv" and os.path.exists(p):
            sys.exit(f"{p} exists: a review may be under way there; use another folder")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", help="dataset table (build_dataset.py labels.csv)")
    ap.add_argument("out_dir")
    ap.add_argument("--splits", default="", help="comma list of splits to include (default: all)")
    ap.add_argument("--sources", default="", help="comma list of sources to include (default: all)")
    ap.add_argument("--skip-sources", default="", help="comma list of sources to leave out")
    args = ap.parse_args()
    cut = lambda s: {x.strip() for x in s.split(",") if x.strip()} or None  # noqa: E731
    rows = make_review_set(args.labels, args.out_dir, cut(args.splits), cut(args.sources), cut(args.skip_sources))
    vis = sum(r["label"] == "present" for r in rows)
    print(f"{len(rows)} frames ({vis} with a visible vulva) -> {args.out_dir}\n"
          f"next: python vulva/label_ellipse.py {args.out_dir} --review <tag>   (only the visible ones are shown)")


if __name__ == "__main__":
    main()
